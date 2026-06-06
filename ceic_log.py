#%%
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
    sigma = 1/5.2  
    gamma = 1/10   
    sigma_t = np.full(total_days, sigma)
    gamma_t = np.full(total_days, gamma)
    params = sigma_t, gamma_t
    return params

#%%
# define NN model architecture and forward pass
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
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

def ode_loss(model, t, params, epsilon, causal = False):
    sigma_array, gamma_array = params
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
    sigma = torch.tensor(sigma_array, dtype=torch.float32)
    gamma = torch.tensor(gamma_array, dtype=torch.float32)
    #force of infection term
    Trans = beta * S * I
    
    # compute the residuals
    r_S = dSdt + Trans
    r_E = dEdt - (Trans - sigma * E)
    r_I = dIdt - (sigma * E - gamma * I)
    r_R = dRdt - (gamma * I)
    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2)

    if causal:
        past_errors = loss.detach()
        #compute number of points
        n_points = past_errors.shape[0]
        cumulative_past_errors = torch.zeros(n_points)
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

#%%
#Initial condition loss
def ic_loss(model, ICs):

    S0, E0, I0, R0 = ICs
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    out = forward_with_constraints(model, t0)
    
    # Value loss - keep gradients!
    pred = out[0, :4]  # selects columns 0, 1, 2, 3, corresponding to S, E, I, R at t =0
    target = torch.tensor([S0, E0, I0, R0], dtype=torch.float32)
    v_loss = ((pred - target)**2).mean()
    return v_loss

#%%
# data loss
def data_loss(model, t_np, I_obs, sigma_array):
    t_k = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    # observed incidence or cumualtive incidence
    observed = torch.tensor((I_obs), dtype=torch.float32)
    
    # data loss 
    # calculate cumulative incidence from the model 
    out = forward_with_constraints(model, t_k)
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    beta = out [:, 4]
    sigma = torch.tensor(sigma_array, dtype=torch.float32)
    predicted_incidence = sigma * E
    v_loss = torch.mean((predicted_incidence - observed) **2)
    # ones = torch.ones_like(E)
    # dEdt = grad.grad(E, t_k, grad_outputs=ones, retain_graph= True)[0].squeeze()
    # C =  beta * S * I - dEdt
    return v_loss


#%%
#Time to train the model
def time_to_train():
    cases = get_syn_data()
    total_days = len(cases)
    t_np_full = np.linspace(0, total_days, total_days, dtype=np.float32)
    t_full = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    return t_np_full, t_full, total_days
#%%
#Train model
def train_model(epochs=10000, test_days=65, causal = False, epsilon = 3, save = False, model_name = 'pinn_model'):
    # general parameters used in the model
    N = 100000
    ICs = [(N-2)/N, 0, 2/N, 0] # no covid cases at the start, seed with 1 or 2 or 10
    cases = get_syn_data()
    Ir_obs = cases / N # convert cases to a proportion
    
    # get time vector, np and tensor types
    t_np_full, t_full, total_days = time_to_train()
    
    # Split into train and test
    train_days = total_days - test_days
    train_indices = np.arange(0, train_days)
    
    # Training time tensor (only training points)
    t_train_np = t_np_full[train_indices] # training time
    t_train_tensor = torch.tensor(t_train_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True) 
    Ir_train = Ir_obs[train_indices] # and cases at those time points

    print(f"t_train_np: {t_train_np[:20].flatten()}")
    print(f"t_train_tensor: {t_train_tensor[:20].flatten()}")

    # if t_train_np (or t_train_tensor) is used to evaluate the model
    # how do you determine data loss? 

    #time_varying parameters for the variants
    params_full = get_parama(total_days)
    sigma_array, gamma_array = params_full
    params = sigma_array[train_indices], gamma_array[train_indices] # only training period parameters

    # Create model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_weights(ep):
        w_ic   = 1.0
        w_ode  = 1.0
        w_data = 10.0
        return w_ic, w_ode, w_data
    print(f"{'Ep':>6} {'IC':>12} {'ODE':>12} {'data':>12} {'Total':>12}  S_range       beta_range")
   
    history = []
    for ep in range(epochs):
        w_ic , w_ode, w_data = get_weights(ep)
        optimizer.zero_grad() #reset any previous gradients

        l_ic = ic_loss(model, ICs)
        l_data = data_loss(model, t_train_np, Ir_train, sigma_array[train_indices])
        if causal:
            l_ode = ode_loss(model, t_train_tensor, params, epsilon=epsilon, causal=True)
        else:
            l_ode = ode_loss(model, t_train_tensor, params, epsilon=0.0, causal=False)

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
                out = forward_with_constraints(model, t_full)
                s_range = f"[{out[:,0].min():.4f}, {out[:,0].max():.4f}]"
                b_range = f"[{out[:,4].min():.3f}, {out[:,4].max():.3f}]"
            print(f"{ep:} {l_ic.item():} {l_ode.item():} {l_data.item():} "
                  f"{loss.item():}  {s_range}  {b_range}")
    if save: 
        torch.save(model.state_dict(), f"{model_name}.pth") 
    return model, history, train_days

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
    t_np_full, t_full, total_days = time_to_train()
    with torch.no_grad():
        out = forward_with_constraints(model, t_full)
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    R = out[:, 3]
    pred_beta = out[:, 4]
    true_beta = seasonal_beta(beta0=0.14, A=0.3, T=365, phase=0)(t_np_full)
    plt.figure(figsize=(14, 8)) 
    plt.subplot(2, 1, 1)
    plt.plot(t_np_full, S, label="S(t)")
    plt.plot(t_np_full, E, label="E(t)")
    plt.plot(t_np_full, I, label="I(t)")
    plt.plot(t_np_full, R, label="R(t)")

    plt.subplot(2, 1, 2)
    plt.plot(pred_beta, color="blue")
    plt.plot(true_beta, color="black")
    plt.title("Beta(t)")
    plt.ylabel("Transmission Rate β(t)")
    plt.grid(True)

#%%
#Plot
def plot_all(model, train_days,t_full):
    N = 100000
    t_np_full, t_full, total_days = time_to_train()
    cases = get_syn_data()
    days = np.arange(total_days)

    sigma_array, gamma_array = get_parama(total_days)
   
    # Convert cases to numpy
    cases_np = np.array(cases)
    #Ic_obs   = np.cumsum(cases_np)
    Ic_obs = cases_np
   
    # Model prediction
    with torch.no_grad():
        out = forward_with_constraints(model, t_full).numpy()
   
    S = out[:, 0] * N
    E = out[:, 1] * N
    I = out[:, 2] * N
    R = out[:, 3] * N
    beta = out[:, 4]
    pred_inci = sigma_array * E
   
  # 1. Training History
    """plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()"""

    #  Cumulative cases
    plt.figure(figsize=(12, 5))
    plt.plot(days, pred_inci, 'b-', label='PINN Prediction', linewidth=2)
    plt.plot(days[:train_days], Ic_obs[:train_days], 'g.', label='Training Data', markersize=3)
    plt.plot(days[train_days:], Ic_obs[train_days:], 'r.', label='Test Data', markersize=3)
    plt.axvline(train_days, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    plt.xlabel('Days')
    plt.ylabel('daily Reported Cases')
    plt.title('PINN Cr vs daily Observed Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("reported_daily_cases_split.png")
    plt.close()
   
    # All Compartments
    plt.figure(figsize=(14, 8))
   
    plt.subplot(2, 3, 1)
    plt.plot(days, S/1e6)
    plt.title(f'Susceptible (final: {S[-1]/1e6:.1f}M)')
    plt.xlabel('Days')
    plt.ylabel('Millions')
    plt.grid(True)
   
    plt.subplot(2, 3, 2)
    plt.plot(days, E/1e3)
    plt.title(f'Exposed (peak: {E.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.grid(True)
   
   
    plt.subplot(2, 3, 3)
    plt.plot(days, I/1e3, 'b-', label='PINN') #PREV
    plt.title(f'Reported (peak: {I.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('prevalence (thousands)')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(2, 3, 4)
    plt.plot(days, R/1e6)
    plt.title(f'Recovered (final: {R[-1]/1e6:.1f}M)')
    plt.xlabel('Days')
    plt.ylabel('Millions')
    plt.grid(True)
   
    plt.subplot(2, 3, 5)
    plt.plot(days, beta)
    plt.title(f'Beta (range: {beta.min():.2f}-{beta.max():.2f})')
    plt.xlabel('Days')
    plt.ylabel('Beta')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(days, pred_inci/1e3)
    plt.title(f'Daily Incidence (final: {pred_inci[-1]/1e3:.1f}k)')
    plt.xlabel('Days')
    plt.ylabel('daily cases (thousands)')
    plt.grid(True)
    plt.savefig("all_plots.png")
    plt.close()

#%%
#plot model comparison (simple PINN vs CEIC PINN)
def model_comparison(model1, model2, t):
    t_np_full, t_full, total_days = time_to_train()
    cases = get_canada_data()
    days = np.arange(total_days)
    N = 100000
    sigma_array, gamma_array = get_parama(total_days)

    with torch.no_grad():
        out1 = forward_with_constraints(model1, t).numpy()
        out2 = forward_with_constraints(model2, t).numpy()

        # Calculate daily predictions
        daily_pred1 = sigma_array * out1[:, 1] * N  # sigma * E * N
        daily_pred2 = sigma_array * out2[:, 1] * N
    #daily cases and cumulative cases comparison
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(t_np_full, daily_pred1, label = 'simple PINN')
    plt.plot(t_np_full, daily_pred2, label = 'CEIC PINN')
    plt.plot(t_np_full, cases, label = 'observed')
    plt.title('daily Cases Comparison')
    plt.xlabel('Days')
    plt.ylabel('daily Cases')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_np_full, np.cumsum(daily_pred1), label = 'simple PINN')
    plt.plot(t_np_full, np.cumsum(daily_pred2), label = 'CEIC PINN')
    plt.plot(t_np_full, np.cumsum(cases), label = 'observed')
    plt.title('cumulative Cases Comparison')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Cases')
    plt.legend()
    plt.grid(True)
    plt.savefig("model_comparison_simple_CEIC.png")
    plt.close()


    plt.figure(figsize=(14, 8))
    plt.plot(t_np_full, out1[:, 4], label = 'simple PINN')
    plt.plot(t_np_full, out2[:, 4], label = "CEIC PINN")
    plt.title('Beta Comparison')
    plt.xlabel('Days')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_comparison_simple_CEIC.png")
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.plot(t_np_full, R_t_simple, label = 'simple PINN')
    plt.plot(t_np_full, R_t_ceic, label = "CEIC PINN")
    plt.plot(t_np_full, Rt_mean, label = 'Bayesian R_t')
    plt.fill_between(t_np_full, Rt_lower, Rt_upper, color = 'grey', alpha = 0.3, label = '95% CI (Bayesian)')
    plt.title('R_t Comparison')
    plt.xlabel('Days')
    plt.ylabel('R_t')
    plt.legend()
    plt.grid(True)
    plt.savefig("R_t_comparison_simple_CEIC.png")
    plt.close()

#Validation metrics table
    print(f"Validation Metrics (last 65 days):")
    print(f"{'Metric':<12} {'CEIC PINN':>18} {'Simple PINN':>18}")
    pred_test1 = daily_pred1[total_days - 65 : total_days]
    pred_test2 = daily_pred2[total_days - 65 : total_days]
    obs_test = np.array(cases)[total_days - 65 : total_days]
    mse1 = np.mean((pred_test1 - obs_test) ** 2)
    mse2 = np.mean((pred_test2 - obs_test) ** 2)
    mae1 = np.mean(np.abs(pred_test1 - obs_test))
    mae2 = np.mean(np.abs(pred_test2 - obs_test))
    #correlation
    corr1 = np.corrcoef(pred_test1, obs_test)[0, 1]
    corr2 = np.corrcoef(pred_test2, obs_test)[0, 1]
    print(f"{'MSE':<12} {mse2:>18.2f} {mse1:>18.2f}")
    print(f"{'MAE':<12} {mae2:>18.2f} {mae1:>18.2f}")
    print(f"{'Correlation':<12} {corr2:>18.4f} {corr1:>18.4f}")

#%%
def load_model(path):
    model = create_model()
    model.load_state_dict(torch.load(path))
    return model

#%%
#Run everything
if __name__ == "__main__":
    N = 100000
    num_runs = 5
    #store errors for each run
    simple_error = []
    ceic_error = []

    #store beta for each run to see the variability across runs
    all_beta_simple = []
    all_beta_ceic = []

    all_C_simple = []
    all_C_ceic = []
    
    for run in range(num_runs):
        t_np_full, t_full, total_days = time_to_train()
        t_full_grad = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

        cases = get_syn_data()
        params = get_parama(total_days)
        sigma_array, gamma_array = params
        print("RUN:", run +1)
        #set_seed(run)
        
        # print("Training simple PINN__")
        # model_simple, history_s, train_days = train_model(epochs = 20000, test_days= 65, causal= False,
        #                                                  epsilon = 3, save = True, model_name= f"simple_pinn_run_{run+1}")
        # print("Training CEIC PINN__")
        # model_ceic, history_c, train_days = train_model(epochs= 20000, test_days= 65, causal = True,
        #                                                epsilon = 3, save = True, model_name= f'ceic_pinn_run_{run+1}')
        # C_simple = compute_C_save(model_simple, t_full_grad, sigma_array, N, f"C_simple_run_{run+1}.npy")
        # C_ceic = compute_C_save(model_ceic,   t_full_grad, sigma_array, N, f"C_ceic_run_{run+1}.npy")
        #load model
        model_simple = load_model(f"simple_pinn_run_{run+1}.pth")
        model_ceic = load_model(f"ceic_pinn_run_{run+1}.pth")
        C_simple = np.load(f"C_simple_run_{run+1}.npy")
        C_ceic = np.load(f"C_ceic_run_{run+1}.npy")

        with torch.no_grad():
            out_simple = forward_with_constraints(model= model_simple, t = t_full).numpy()
            out_ceic = forward_with_constraints(model= model_ceic, t = t_full).numpy()

            beta_simple = out_simple[:, 4]  # beta is column 4
            beta_ceic   = out_ceic[:, 4]

            all_beta_simple.append(beta_simple)
            all_beta_ceic.append(beta_ceic)

            all_C_simple.append(C_simple)
            all_C_ceic.append(C_ceic)
            
        #model_comparison(model_simple, model_ceic, t_full)

            #calling observed beta
        true_beta = data_gen.seasonal_beta(beta0=0.3, A=0.2, T=180, phase=0)(t_np_full)
        #true_beta = data_gen.piecewise_beta([0.1, 0.15, 0.25, 0.3, 0.4], [60,120, 240, 300])(t_np_full)
        #plot beta variability across runs for CEIC and true beta and simple PINN
        plt.figure(figsize=(14, 8))
        for i, beta in enumerate(all_beta_ceic):
            plt.plot(t_np_full, beta, color='blue', label='CEIC PINN')
        plt.plot(t_np_full, true_beta, label = 'True Beta', color = 'red')
        plt.title('Beta Variability Across Runs')
        plt.xlabel('Days')
        plt.ylabel('Beta')
        plt.legend()
        plt.savefig("beta_variability_and_ceic.png")
        plt.close()

        plt.figure(figsize=(14, 8))
        for i, beta in enumerate(all_beta_simple):
            plt.plot(t_np_full, beta, color='green', label='Simple PINN')
        plt.plot(t_np_full, true_beta, label = 'True Beta', color = 'red')
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Beta')
        plt.savefig("beta_variability_and-simple.png")
        plt.close()

        #plot the observed data with simple and ceic prediction
        plt.figure(figsize=(14, 8))
        for i, C_simple in enumerate(all_C_simple):
            plt.plot(t_np_full, C_simple * N, label = 'simple pinn', color = 'orange')
        plt.plot(t_np_full, cases, label = 'observed', color = 'green')
        plt.legend()
        plt.savefig("comparison of Observed data , simple")
        plt.close()

        plt.figure(figsize=(14, 8))
        for i, C_ceic in enumerate(all_C_ceic):
            plt.plot(t_np_full, C_simple * N, label = 'ceic pinn', color = 'blue')
        plt.plot(t_np_full, cases, label = 'observed', color = 'green')
        plt.legend()
        plt.savefig("comparison of Observed data , ceic")
        plt.close()
        

        


            

        



    

