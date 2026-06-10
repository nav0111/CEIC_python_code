from curses import raw
import datetime
import itertools
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
from data_gen import generate_data_with_defaults, seasonal_beta

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_syn_data():
    # get the incidence curve and the beta curve
    betafn = seasonal_beta(beta0=0.14, A=0.3, T=365, phase=0)
    data3, beta3 = generate_data_with_defaults(betafn) 
    return data3, beta3 

#synthetic data params
def get_parama():
    """Constant sigma and gamma for synthetic data"""
    sigma_t = 1/5.2
    gamma_t = 1/10
    omega_t = 1/60
    params = sigma_t, gamma_t, omega_t
    return params

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
    beta = 0.6 * torch.sigmoid(raw[:, 4:5]) 
    return torch.cat([S, E, I, R, beta], dim=1)

def ode_loss(model, tk, max_time, epsilon, causal = False):
    out = forward_with_constraints(model, tk)
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    R = out[:, 3]
    beta = out[:, 4]

    ones = torch.ones_like(S) #compute derivative for each point independently. 

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
    dSdt = grad.grad(S, tk, grad_outputs=ones, create_graph = True)[0].squeeze()
    dEdt = grad.grad(E, tk, grad_outputs=ones, create_graph = True)[0].squeeze()
    dIdt = grad.grad(I, tk, grad_outputs=ones, create_graph = True)[0].squeeze()
    dRdt = grad.grad(R, tk, grad_outputs=ones, create_graph = True)[0].squeeze()
   
    #convert sigma and gamma arrays to tensors and get the values at the corresponding time points
    sigma = 1/5.2
    gamma = 1/10
    omega = 1/60
    
    #force of infection term
    Trans = beta * S * I
    
    # compute the residuals
    r_S = dSdt + max_time * (Trans - (omega * R))
    r_E = dEdt - max_time * (Trans - sigma * E)
    r_I = dIdt - max_time * (sigma * E - gamma * I)
    r_R = dRdt - max_time * (gamma * I - omega * R)
    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2)

    # CRITICAL: Divide the raw squared residuals by max_time**2 
    # This normalizes the continuous physics loss back to O(1) bounds
    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2) / (max_time ** 2)

    if causal:
        # CRITICAL: Normalize the loss to O(1) before passing to exponent
        # Otherwise exp(-epsilon * 100000) underflows to 0 instantly
        normalized_loss = loss.detach()
        n_points = normalized_loss.shape[0]
        cumulative_past_errors = torch.zeros(n_points, device=device)
        cumulative_past_errors[1:] = torch.cumsum(normalized_loss[:-1], dim=0)

        weights = torch.exp(-epsilon * cumulative_past_errors)
        weights[0] = 1.0
        
        # Apply weights to the ORIGINAL loss for gradient updates
        ode_loss = (weights * loss).mean()
    else:
        ode_loss = loss.mean()

    return ode_loss
    

def ic_loss(model, ICs, t0):
    # evaluates the model at t = 0
    S0, E0, I0, R0 = ICs
    out = forward_with_constraints(model, t0)
    pred = out[0, :4]  # selects columns 0, 1, 2, 3, corresponding to S, E, I, R at t =0
    target = torch.tensor([S0, E0, I0, R0], dtype=torch.float32, device=device)
    v_loss = ((pred - target)**2).mean()
    return v_loss

def data_loss(model, tk, observed):
    # data loss, run the model on the time tensor
    out = forward_with_constraints(model, tk)
    E = out[:, 1]
    sigma = 1/5.2 # replace with get_parama later.
    predicted_incidence = sigma * E
    v_loss = torch.mean((predicted_incidence - observed)**2)
    return v_loss

def prev_loss(model, tk, observed):
    out = forward_with_constraints(model, tk)
    I = out[:,2]
    v_loss = torch.mean(( I - observed)**2)
    return v_loss

def conservation_loss(model, t):
    out = forward_with_constraints(model, t)
    total = out[:, 0] + out[:, 1] + out[:, 2] + out[:, 3]
    return torch.mean((total - 1.0) ** 2)

def time_to_train(total_days = 730):
    # create time tensors, normalized from 0 to 1. 
    t_data_tensor = torch.linspace(0, 1, steps=total_days, device=device, dtype=torch.float32).reshape(-1, 1).requires_grad_()
    t_colloc_tensor = torch.linspace(0, 1, steps=5000, device=device, dtype=torch.float32).reshape(-1, 1).requires_grad_()
    t_ic_tensor = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    return t_data_tensor, t_colloc_tensor, t_ic_tensor

def train_model(epochs=6000, causal = False, epsilon = 3, save = False, model_name = 'pinn_model'):
    # general parameters used in the model
    N = 1000
    ICs = [(N-5)/N, 0, 5/N, 0] # no covid cases at the start, seed with 1 or 2 or 10
    cases, _ = get_syn_data()
    sigma, gamma, omega = get_parama()
    Ir_obs = cases / N # convert cases to a proportion
    obs_tensor = torch.tensor((Ir_obs), dtype=torch.float32, device=device) 
   
    # get time vector, np and tensor types\
    # plus a t0 tensor to pass to the IC loss function, 
    # since it needs to evaluate the model at t=0
    max_time = len(Ir_obs)
    t_data_tensor, t_colloc_tensor, t_ic_tensor = time_to_train(max_time)
 
    # Create model
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_weights(ep):
        if ep < 1000:
            return 100.0, 1.0, 1.0   # data-driven warmup, no physics
        elif ep < 3000:
            return 100.0, 100.0, 100.0   # gradually re-introduce physics
        else:
            return 100.0, 2000.0, 1000.0   # full training
    
    history = []    
    logs = []   # store rows here
    bar_len = 30 # create a little pbar 

    for ep in range(epochs):
        optimizer.zero_grad() #reset any previous gradients
        w_ic, w_ode, w_data = get_weights(ep)  # initialize weights
        l_ic = ic_loss(model, ICs, t_ic_tensor)
        #l_data = data_loss(model, t_data_tensor, obs_tensor)
        l_data = prev_loss(model, t_data_tensor, obs_tensor)
        l_cons = conservation_loss(model, t_colloc_tensor) 
        if causal:
            l_ode = ode_loss(model, t_colloc_tensor, max_time, epsilon=epsilon, causal=True)
        else:
            l_ode = ode_loss(model, t_colloc_tensor, max_time, epsilon=0.0, causal=False)
        loss = w_ic * l_ic  + w_ode * l_ode + w_data * l_data + 500.0 * l_cons

        loss.backward()
        
        # Gradient clipping
        #calculate L2 norm for all gradients, if its >1 then scale it to norm=1, and if <1 then does nothing
        #we need this to be safe from exploding gradient
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
       
        # detach the computation graph from loss
        history.append(loss.detach()) 
        
        # save logs every 1000 epochs
        if ep % 1000 == 0:
            # at every 1000, run the model to see how well it's performing. 
            progress = int((ep / epochs) * bar_len)
            bar = "#" * progress + "-" * (bar_len - progress)
            print(f"\r[{bar}] {ep}/{epochs}", end="")          

            with torch.no_grad():
                out = forward_with_constraints(model, t_data_tensor)
                # get the susceptible, beta vector min/max 
                s_min, s_max = out[:,0].min(), out[:,0].max()
                b_min, b_max = out[:,4].min(), out[:,4].max()
            # store GPU tensors directly (no .item())
            logs.append([
                ep,
                l_ic.detach(),
                l_ode.detach(),
                l_data.detach(),
                l_cons.detach(),
                loss.detach(),
                s_min, s_max,
                b_min, b_max
            ])

    # print out logging. 
    print("finished training")
    print(f"{'ep':>8} {'l_ic':>10} {'l_ode':>10} {'l_data':>10} {'l_cons':>10} {'loss':>10} "
      f"{'s_range':<18} {'b_range':<18}")
    for row in logs:
        ep, l_ic, l_ode, l_data, l_cons, loss, s_min, s_max, b_min, b_max = row
        
        print(
            f"{ep:8d} "
            f"{l_ic.item():12.3e} "
            f"{l_ode.item():12.3e} "
            f"{l_data.item():12.3e} "
            f"{l_cons.item():12.3e} "
            f"{loss.item():12.3e} "
            f"[{s_min.item():.3f}, {s_max.item():.3f}]{'':>3}"
            f"[{b_min.item():.3f}, {b_max.item():.3f}]"
        )
    history = [h.cpu().item() for h in history]

    
    with torch.no_grad():
        out = forward_with_constraints(model, t_data_tensor)
        pred_inci = out[:, 2].cpu().numpy()
    print("\nFinal Predictions on Training Data:")    
    print(f"Predicted incidence range: [{pred_inci.min():.3e}, {pred_inci.max():.3e}]")
    print(f"Predicted incidence mean:  {pred_inci.mean():.3e}")
    print(f"Observed incidence range:  [{Ir_obs.min():.3e}, {Ir_obs.max():.3e}]")
    print(f"Observed incidence mean:   {Ir_obs.mean():.3e}")
    print(f"'Predict zero' floor:      {(Ir_obs**2).mean():.3e}")
    print(f"Actual l_data:             {l_data.item():.3e}")

    if save: 
        torch.save(model.state_dict(), f"{model_name}.pth") 
    return model, history

def load_model(path):
    model = create_model().to(device)
    model.load_state_dict(torch.load(path))
    return model

def plot_loss(l_history, fname = "training_history.png"):
    plt.figure(figsize=(10, 4))
    
    plt.title('Training History')
    plt.grid(True)
    plt.savefig(f"{fname}_loss.png")
    #plt.close()

def get_nn_interpolated_beta(model, time_tensor):
    with torch.no_grad():
         out = forward_with_constraints(model, time_tensor)
    pred_beta = out[:, 4].cpu().detach().numpy()
    pred_beta[0] = 0.15
    #print(pred_beta[: 5])
    #t_array = t_data_tensor.detach().numpy().reshape(-1)
    # or 
    t_array = time_tensor.detach().squeeze().numpy()
    beta_fn = interp1d(t_array, pred_beta, kind='cubic')
    return beta_fn 


def plot_model(model, loss_history, fname = "model_predictions.png"):
    obs, true_beta = get_syn_data()
    t_data_tensor, _, _ = time_to_train()
    print(t_data_tensor[:5])
    # evaluate the trained model on the training points
    # but these are between 0 and 1? 
    multiplier = 1000 

    with torch.no_grad():
        out = forward_with_constraints(model, t_data_tensor)
    S = out[:, 0].cpu().numpy() * multiplier
    E = out[:, 1].cpu().numpy() * multiplier
    I = out[:, 2].cpu().numpy() * multiplier
    R = out[:, 3].cpu().numpy() * multiplier
    pred_beta = out[:, 4].cpu().numpy()

    
    # test the interpolation function for beta
    # time tensor between 0 and 730
    t_tensor = torch.linspace(0, 730, steps = 730, device=device, dtype=torch.float32).reshape(-1, 1)
    beta_fn = get_nn_interpolated_beta(model, t_tensor)
    plt.figure(figsize=(10, 4))
    t_test = np.linspace(0, 1.01, 730)
    pred_beta_test = beta_fn(t_test)
    #print(pred_beta_test)
    plt.plot(t_test, pred_beta_test, label="Interpolated β(t)")


    #computing the parameters of the predictive curve
    beta_mean = pred_beta.mean()
    beta_min = pred_beta.min()
    beta_max= pred_beta.max()

    #getting data using the predictive curve
    data_nn, _ = generate_data_with_defaults(beta_fn)
    #print(f"NN Data: {data_nn}")
    

    

    # # check model consistency
    total = out[:, :4].sum(dim=1)
    print(f"Conservation: min={total.min():.4f}, max={total.max():.4f}")

    print(f"β range: [{pred_beta.min():.4f}, {pred_beta.max():.4f}]")
    print(f"β mean: {pred_beta.mean():.4f}")
    print(f"True β range: [0.098, 0.182] (for seasonal 0.14 ± 30%)")

    print(f"S range: [{out[:,0].min():.4f}, {out[:,0].max():.4f}]")
    print(f"S at end: {out[-1, 0]:.4f}")
    #print(out[:, 2] * 1000)
    
    
    plt.figure(figsize=(14, 8)) 
    plt.subplot(4, 1, 1)
    plt.plot(S, label="S(t)")
    plt.plot(E, label="E(t)")
    plt.plot(I, label="I(t)")
    plt.plot(R, label="R(t)")
    plt.ylabel("SEIR model")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(pred_beta, color="blue")
    plt.plot(true_beta, color="black")
    plt.title("Beta(t)")
    plt.ylabel("Transmission Rate β(t)")
    plt.grid(True)
    

    plt.subplot(4, 1, 3)
    plt.plot(obs, label="Observed Incidence")
    plt.plot(I, label="modelled I(t)")
    plt.ylabel("modelled vs observed I")
    plt.plot(data_nn, label = 'nn_data')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(loss_history, label="Total Loss")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig(f"{fname}_check.png")
    plt.close()
    

def testm(): 
    is_causal_options = [False, True]  # False = Vanilla, True = Causal
    
    #for causal, w_ic, w_data in itertools.product(is_causal_options, ic_weights, data_weights):
    for causal in is_causal_options:
        model_type = "causal" if causal else "vanilla"
        fname = f"{model_type}_icx_datax"
        print(f"Starting Run: {fname}")
        print(f"{'='*10}")
        m1, h1 = train_model(
            epochs = 10000,
            save = True, 
            model_name = fname, 
            causal = causal, 
            epsilon = 0.5 if causal else 0.0, 
        )
        plot_model(m1, h1, fname) 

#testm()
simple_model = load_model("vanilla_icx_datax.pth")
plot_model(simple_model, [], fname = "vanilla_icx_datax_final")
ceic_model = load_model("causal_icx_datax.pth")
plot_model(ceic_model, [], fname= "causal_icx_datax_final" )



    

   


    