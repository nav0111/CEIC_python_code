import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as grad
import torch.nn.functional as F
from scipy import stats
from scipy.stats import gamma as gamma_dist
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import random
from data_gen import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_syn_data():
    beta_fn3 = seasonal_beta(beta0=0.14, A=0.3, T=365, phase=0)
    data3 = generate_data_with_defaults(beta_fn3) 
    return data3 # let's play with the seasonal one. 

#synthetic data params
def get_parama(total_days):
    """Constant sigma and gamma for synthetic data"""
    sigma_t = np.full(total_days, 1/5.2)
    gamma_t = np.full(total_days, 1/10)
    omega_t = np.full(total_days, 1/60) 
    params = sigma_t, gamma_t, omega_t
    return params

def create_model():
    model = nn.Sequential(
        nn.Linear(1, 96), nn.SiLU(),
        nn.Linear(96, 96), nn.SiLU(),
        nn.Linear(96, 96), nn.SiLU(),
        nn.Linear(96, 5)
    )
    return model

def forward_with_constraints(model, t):
    raw = model(t)
    S = torch.sigmoid(raw[:, 0:1])
    E = F.softplus(raw[:, 1:2])
    I = F.softplus(raw[:, 2:3])
    R = F.softplus(raw[:, 3:4])
    beta = F.softplus(raw[:, 4:5])
    return torch.cat([S, E, I, R, beta], dim=1)

def ode_loss(model, t, epsilon, causal = False):
    out = forward_with_constraints(model, t)
   
    # the output of the neural network at time t
    # since t is a tensor, S is also a tensor and stores the computation graph
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    R = out[:, 3]
    beta = out[:, 4]

   #creates a tensor of all 1's same shape as S, need this to combine gradients from multiple outputs
    ones = torch.ones_like(S) #compute derivative for each point independently. 
    #If E = [E1, E2, E3, ...] (vector)
    #grad_outputs = [v1, v2, v3, ...]
    #Then result = v1 * ∂E1/∂t + v2 * ∂E2/∂t + v3 * ∂E3/∂t + ...
    #create_graph= True, because we need derivative of a derivative otherwise the term would consider as a constant
   
    # the tensor ones is needed because grad computes VJP. 
    # retain_graph is needed for the repeated grad calls after S (for E, I, R)
    # create_graph is needed, otherwise torch treats the gradients as constants? 
    # meaning, it computes dSdt numerically, but it does not build a computation graph 
    # that connects dSdt back to the model parameters
    # meaning dSdt becomes a leaf tensor: it has grad_fn=None, and no gradients can flow through it when you later compute ode_loss.backward().
    # in a PINN, we want dS/dt as a differentiable function of the network parameters since 
    # the loss function is built from dS/dT. 
    # here is an easier way to understand: Let Q = dS/dt and we have L = Q + beta*s*I 
    # and ultimately in backprop we need to do L' which means we need Q' (w.r.t to the parameters)
    # which means we need to be able to do dQ/dtheta, which is d(dS/dt)/dtheta, which is the second derivative of S w.r.t time and parameters.
    # so create_graph creats the computational graph so we can do ode_loss.backward() later on
    # squeeze: giving you a one-dimensional tensor of shape (n_points,) instead of (n_points, 1)
    dSdt = grad.grad(S, t, grad_outputs=ones, create_graph = True, retain_graph= True)[0].squeeze()
    dEdt = grad.grad(E, t, grad_outputs=ones, create_graph = True, retain_graph= True)[0].squeeze()
    dIdt = grad.grad(I, t, grad_outputs=ones, create_graph = True, retain_graph= True)[0].squeeze()
    dRdt = grad.grad(R, t, grad_outputs=ones, create_graph = True, retain_graph= True)[0].squeeze()
   
    #convert sigma and gamma arrays to tensors and get the values at the corresponding time points
    sigma = 1/5.2
    gamma = 1/10
    omega = 1/60
    
    #force of infection term
    Trans = beta * S * I
    
    # compute the residuals
    r_S = dSdt + Trans - (omega * R)
    r_E = dEdt - (Trans - sigma * E)
    r_I = dIdt - (sigma * E - gamma * I)
    r_R = dRdt - (gamma * I - omega * R)
    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2)

    if causal:
        past_errors = loss.detach()
        #compute number of points
        n_points = past_errors.shape[0]
        cumulative_past_errors = torch.zeros(n_points, device=device)
        cumulative_past_errors[1:] = torch.cumsum(past_errors[:-1], dim = 0)

    #weight = exp(-epsilon * cumulative past errors), very small ϵcan prevent the network 
    # from effectively minimizing the latter temporal residuals, a large ϵ value can result in a more difficult optimization problem, because the temporal residuals at earlier times have to decrease to a very small value in order to activate the latter temporal weights
        weights = torch.exp(-epsilon * cumulative_past_errors)

    #at t=0, weight must be 1, because we should learn the IC first
        weights[0] = 1.0
        weighted_error = weights * loss

        ode_loss = weighted_error.mean()
    else:

        ode_loss = loss.mean()

    return ode_loss

def ic_loss(model, ICs):
    S0, E0, I0, R0 = ICs
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    out = forward_with_constraints(model, t0)
    
    # Value loss - keep gradients!
    pred = out[0, :4]  # selects columns 0, 1, 2, 3, corresponding to S, E, I, R at t =0
    target = torch.tensor([S0, E0, I0, R0], dtype=torch.float32, device=device)
    v_loss = ((pred - target)**2).mean()
    return v_loss

def data_loss(model, t_np, I_obs):
    t_k = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    # observed incidence or cumualtive incidence
    observed = torch.tensor((I_obs), dtype=torch.float32, device=device)
    
    # data loss 
    # calculate cumulative incidence from the model 
    out = forward_with_constraints(model, t_k)
    E = out[:, 1]
    sigma = 1/5.2
    predicted_incidence = sigma * E
    v_loss = torch.mean((predicted_incidence - observed) **2)
    return v_loss

def conservation_loss(model, t):
    out = forward_with_constraints(model, t)
    total = out[:, 0] + out[:, 1] + out[:, 2] + out[:, 3]
    return torch.mean((total - 1.0) ** 2)

def time_to_train():
    cases = get_syn_data()
    total_days = len(cases)
    #t_np_full = np.linspace(0, total_days, total_days, dtype=np.float32)
    t_np_full = np.arange(total_days, dtype=np.float32)
    t_full = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

    # Data points (one per day, where observations exist)
    t_data_np = np.arange(total_days, dtype=np.float32)
    t_data_tensor = torch.tensor(t_data_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

    # Collocation points (denser, for ODE residual)
    n_colloc = 5000  # vs ~720 data points
    t_colloc_np = np.linspace(0, total_days - 1, n_colloc).astype(np.float32)
    t_colloc_tensor = torch.tensor(t_colloc_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    return t_data_np, t_data_tensor, t_colloc_np, t_colloc_tensor, total_days

def train_model(epochs=10000, test_days=0, causal = False, epsilon = 3, save = False, model_name = 'pinn_model'):
    # general parameters used in the model
    N = 100000
    ICs = [(N-2)/N, 0, 2/N, 0] # no covid cases at the start, seed with 1 or 2 or 10
    cases = get_syn_data()
    Ir_obs = cases / N # convert cases to a proportion
    
    # get time vector, np and tensor types
    t_data_np, t_data_tensor, t_colloc_np, t_colloc_tensor, total_days = time_to_train()
    
    # # Split into train and test
    # train_days = total_days - test_days
    # train_indices = np.arange(0, train_days)
    
    # # Training time tensor (only training points)
    # t_train_np = t_data_np[train_indices] # training time
    # t_train_tensor = torch.tensor(t_train_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True) 
    # Ir_train = Ir_obs[train_indices] # and cases at those time points

    # print(f"t_train_np: {t_train_np[:20].flatten()}")
    # print(f"t_train_tensor: {t_train_tensor[:20].flatten()}")

    # if t_train_np (or t_train_tensor) is used to evaluate the model
    # how do you determine data loss? 

    # Create model
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_weights(ep):
        w_ic   = 1.0
        w_ode  = 1.0
        w_data = 1.0
        return w_ic, w_ode, w_data
    print(f"{'Ep':>6} {'IC':>12} {'ODE':>12} {'data':>12} {'Total':>12}  S_range       beta_range")
   
    history = []
    for ep in range(epochs):
        w_ic , w_ode, w_data = get_weights(ep)
        optimizer.zero_grad() #reset any previous gradients

        l_ic = ic_loss(model, ICs)
        l_data = data_loss(model, t_data_np, Ir_obs)
        if causal:
            l_ode = ode_loss(model, t_colloc_tensor, epsilon=epsilon, causal=True)
        else:
            l_ode = ode_loss(model, t_colloc_tensor, epsilon=0.0, causal=False)

        loss = w_ic * l_ic + w_ode * l_ode + w_data * l_data
        loss.backward()
        
        # Gradient clipping
        #calculate L2 norm for all gradients, if its >1 then scale it to norm=1, and if <1 then does nothing
        #we need this to be safe from exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
       
        history.append(loss.item()) #loss is a tensor with attached graph, it remembers how it was computed,
        #takes lot of memory, we need to convert it to a scalar value so 
        # we detach it from the graph and then convert to a python number using item()
       
        if ep % 1000 == 0:
            with torch.no_grad():
                out = forward_with_constraints(model, t_data_tensor)
                s_range = f"[{out[:,0].min():.4f}, {out[:,0].max():.4f}]"
                b_range = f"[{out[:,4].min():.3f}, {out[:,4].max():.3f}]"
            print(f"{ep:} {l_ic.item():} {l_ode.item():} {l_data.item():} "
                  f"{loss.item():}  {s_range}  {b_range}")
    
    print("finished training")
    if save: 
        torch.save(model.state_dict(), f"{model_name}.pth") 
    return model, history


def load_model(path):
    model = create_model().to(device)
    model.load_state_dict(torch.load(path))
    return model

def plot_loss(l_history):
    plt.figure(figsize=(10, 4))
    plt.plot(l_history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True)
    plt.savefig("training_history.png")
    #plt.close()

def plot_model(model):
    t_data_np, t_data_tensor, t_colloc_np, t_colloc_tensor, total_days = time_to_train()
    #t_np_full, t_full, total_days = time_to_train()
    with torch.no_grad():
        out = forward_with_constraints(model, t_data_tensor)
    S = out[:, 0].cpu().numpy()
    E = out[:, 1].cpu().numpy()
    I = out[:, 2].cpu().numpy()
    R = out[:, 3].cpu().numpy()
    pred_beta = out[:, 4].cpu().numpy()

    # check model consistency
    total = out[:, :4].sum(dim=1)
    print(f"Conservation: min={total.min():.4f}, max={total.max():.4f}")

    print(f"β range: [{pred_beta.min():.4f}, {pred_beta.max():.4f}]")
    print(f"β mean: {pred_beta.mean():.4f}")
    print(f"True β range: [0.098, 0.182] (for seasonal 0.14 ± 30%)")

    print(f"S range: [{out[:,0].min():.4f}, {out[:,0].max():.4f}]")
    print(f"S at end: {out[-1, 0]:.4f}")

    true_beta = seasonal_beta(beta0=0.14, A=0.3, T=365, phase=0)(t_data_np)
    plt.figure(figsize=(14, 8)) 
    plt.subplot(2, 1, 1)
    plt.plot(t_data_np, S, label="S(t)")
    plt.plot(t_data_np, E, label="E(t)")
    plt.plot(t_data_np, I, label="I(t)")
    plt.plot(t_data_np, R, label="R(t)")

    plt.subplot(2, 1, 2)
    plt.plot(pred_beta, color="blue")
    plt.plot(true_beta, color="black")
    plt.title("Beta(t)")
    plt.ylabel("Transmission Rate β(t)")
    plt.grid(True)
    plt.show()

def testm(): 
    print("testing all models")
    m1, h1 = train_model() 
    plot_loss(h1)
    plot_model(m1)
    #m1, h1 = train_model(causal = True, epsilon = 3) 
    #plot_loss(h1)
    #plot_model(m1)
    m1, h1 = train_model(causal = True, epsilon = 0.5) 
    plot_loss(h1)
    plot_model(m1)


# #Run everything
# if __name__ == "__main__":
#     N = 100000
#     num_runs = 5
#     #store errors for each run
#     simple_error = []
#     ceic_error = []

#     #store beta for each run to see the variability across runs
#     all_beta_simple = []
#     all_beta_ceic = []

#     all_C_simple = []
#     all_C_ceic = []
    
#     for run in range(num_runs):
#         t_np_full, t_full, total_days = time_to_train()
#         t_full_grad = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

#         cases = get_syn_data()
#         params = get_parama(total_days)
#         sigma_array, gamma_array = params
#         print("RUN:", run +1)
#         #set_seed(run)
        
#         # print("Training simple PINN__")
#         # model_simple, history_s, train_days = train_model(epochs = 20000, test_days= 65, causal= False,
#         #                                                  epsilon = 3, save = True, model_name= f"simple_pinn_run_{run+1}")
#         # print("Training CEIC PINN__")
#         # model_ceic, history_c, train_days = train_model(epochs= 20000, test_days= 65, causal = True,
#         #                                                epsilon = 3, save = True, model_name= f'ceic_pinn_run_{run+1}')
#         # C_simple = compute_C_save(model_simple, t_full_grad, sigma_array, N, f"C_simple_run_{run+1}.npy")
#         # C_ceic = compute_C_save(model_ceic,   t_full_grad, sigma_array, N, f"C_ceic_run_{run+1}.npy")
#         #load model
#         model_simple = load_model(f"simple_pinn_run_{run+1}.pth")
#         model_ceic = load_model(f"ceic_pinn_run_{run+1}.pth")
#         C_simple = np.load(f"C_simple_run_{run+1}.npy")
#         C_ceic = np.load(f"C_ceic_run_{run+1}.npy")

#         with torch.no_grad():
#             out_simple = forward_with_constraints(model= model_simple, t = t_full).numpy()
#             out_ceic = forward_with_constraints(model= model_ceic, t = t_full).numpy()

#             beta_simple = out_simple[:, 4]  # beta is column 4
#             beta_ceic   = out_ceic[:, 4]

#             all_beta_simple.append(beta_simple)
#             all_beta_ceic.append(beta_ceic)

#             all_C_simple.append(C_simple)
#             all_C_ceic.append(C_ceic)
            
#         #model_comparison(model_simple, model_ceic, t_full)

#             #calling observed beta
#         true_beta = data_gen.seasonal_beta(beta0=0.3, A=0.2, T=180, phase=0)(t_np_full)
#         #true_beta = data_gen.piecewise_beta([0.1, 0.15, 0.25, 0.3, 0.4], [60,120, 240, 300])(t_np_full)
#         #plot beta variability across runs for CEIC and true beta and simple PINN
#         plt.figure(figsize=(14, 8))
#         for i, beta in enumerate(all_beta_ceic):
#             plt.plot(t_np_full, beta, color='blue', label='CEIC PINN')
#         plt.plot(t_np_full, true_beta, label = 'True Beta', color = 'red')
#         plt.title('Beta Variability Across Runs')
#         plt.xlabel('Days')
#         plt.ylabel('Beta')
#         plt.legend()
#         plt.savefig("beta_variability_and_ceic.png")
#         plt.close()

#         plt.figure(figsize=(14, 8))
#         for i, beta in enumerate(all_beta_simple):
#             plt.plot(t_np_full, beta, color='green', label='Simple PINN')
#         plt.plot(t_np_full, true_beta, label = 'True Beta', color = 'red')
#         plt.legend()
#         plt.xlabel('Days')
#         plt.ylabel('Beta')
#         plt.savefig("beta_variability_and-simple.png")
#         plt.close()

#         #plot the observed data with simple and ceic prediction
#         plt.figure(figsize=(14, 8))
#         for i, C_simple in enumerate(all_C_simple):
#             plt.plot(t_np_full, C_simple * N, label = 'simple pinn', color = 'orange')
#         plt.plot(t_np_full, cases, label = 'observed', color = 'green')
#         plt.legend()
#         plt.savefig("comparison of Observed data , simple")
#         plt.close()

#         plt.figure(figsize=(14, 8))
#         for i, C_ceic in enumerate(all_C_ceic):
#             plt.plot(t_np_full, C_simple * N, label = 'ceic pinn', color = 'blue')
#         plt.plot(t_np_full, cases, label = 'observed', color = 'green')
#         plt.legend()
#         plt.savefig("comparison of Observed data , ceic")
#         plt.close()
        

        


            

        



    

