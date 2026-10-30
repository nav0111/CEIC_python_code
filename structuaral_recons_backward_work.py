import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from data_gen import *

#fit a feedforward network to z_obs(t) first
#structural reconstruction using the ODEs
# Z(t) = sigma * E(t) => E(t) = Z(t) / sigma
# I(t) = by solving dI/dt = Z(t) - gamaa * I(t) --- integration
# R(t) = by solving dR/dt = gamma * I(t) - omega * R(t) --- integration
# S(t) = N - E(t) - I(t) - R(t)   (N, not 1 -- data_gen works in raw counts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Base_Dir = os.path.dirname(os.path.abspath(__file__))
Output_Dir = os.path.join(Base_Dir, "structual_reconstruction_backward_results")
os.makedirs(Output_Dir, exist_ok=True)
Train = True

#True synthetic data
def get_true_data():
    beta_fn = seasonal_beta(beta0=0.14, A =0.3, T = 365, phase = 0)
    N, S0, E0, I0, R0, C0 = get_initial_conditions()
    sigma, gamma, omega = get_parama()
    t = np.arange(730, dtype=np.float64)
    S, E, I, R, C, Z = gen_inci(beta_fn, sigma, gamma, omega, S0, E0, I0, R0,C0, t)
    noise = np.random.normal(0.0, 0.1*np.max(Z), size = len(Z))
    Z_obs = np.maximum(0.0, (Z + noise))
    return t, S, E, I, R, C, Z, beta_fn, sigma, gamma, omega, N, Z_obs

#create a feedforward NN to fit the daily incidence data
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )
    return model

#squared output to keep incidence positive
def positive_output(model, t):
    return model(t)**2

#define derivative
def derivative(y,x):
    return torch.autograd.grad(y, x, grad_outputs= torch.ones_like(y),
                               create_graph= True, retain_graph= True)[0]

#####checking the derivative function
# def fn(x):
#     return x**3

# x = torch.tensor([1.0, 2.0, 3.0], requires_grad= True)
# fnn = x**3

def physical_derivative(y, tau, max_time):
    return derivative(y, tau) / max_time

#fitting the NN to the daily incidence data
def fit_incidence_nn(z_obs_tensor, tau_data):
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for epoch in range(1500):
        optimizer.zero_grad()
        z_hat = positive_output(model, tau_data)
        loss = torch.mean((z_hat - z_obs_tensor)**2)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

def return_fitted_model():
    (t, S_true, E_true, I_true, R_true, C_true, Z_true, beta_true,
        sigma, gamma, omega, N, Z_obs) = get_true_data()
    
    t = torch.linspace(0, 1.0, len(Z_true)).reshape(-1, 1)
    z_obs = torch.tensor(Z_obs[:, None])
    nn_model = fit_incidence_nn(z_obs, t)
    return t, nn_model
    

#load saved incidence model
def load_saved_incidence_model():
    path = os.path.join(Output_Dir, "incidence_model.pt")
    model = create_model().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

#The reconstruction equation integration part

# def integrate_linear_ode(dydt, y0, t_span, t_eval):
#     sol = solve_ivp(dydt, t_span, [y0], method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9)
#     return sol.y[0]

# #reconstruction from incidence
# def reconstruction_from_incidence(z_hat, dz_hat_dt, beta_fn, sigma, gamma, omega, N, S0, E0,
#                                   I0, R0, C0, t):
#     e = z_hat / sigma
#     # np.interp is used to interpolate the incidence data for integration
#     i = integrate_linear_ode(lambda tt, I: np.interp(tt, t, z_hat) - gamma * I, I0, (0, len(t) - 1), t)
#     r = integrate_linear_ode(lambda tt, R: gamma * np.interp(tt, t, i) - omega * R, R0, (0, len(t) - 1), t)
#     s = N - e - i - r
#     # beta = N*(dE/dt + sigma*E)/(S*I)
#     beta = N * (dz_hat_dt / sigma + z_hat) / (s * i) 
#     return s, e, i, r, beta

#solve the ODEs
def sovle_odes(z_hat, dz_hat_dt, sigma, gamma, omega, N, I0, R0, t):
    e = z_hat / sigma
    def dIdt(tt,I):
        return np.interp(tt, t, z_hat) - gamma * I

    i_sol = solve_ivp(dIdt, (0, len(t) -1), [I0], t_eval=t, method='RK45', rtol=1e-6, atol=1e-9)

    i = i_sol.y[0]

    def dRdt(tt, R):
        return gamma * np.interp(tt, t, i) - omega * R

    r_sol = solve_ivp(dRdt, (0, len(t) -1), [R0], t_eval=t, method = "RK45", rtol=1e-6, atol=1e-9)

    r = r_sol.y[0]

    s = N - e - i - r

    beta = N * (dz_hat_dt / sigma + z_hat) / (s * i)

    return s, e, i, r, beta

#testing interpolation
def test_interpolation():
    data = get_true_data()
    sigma, gamma, omega = get_parama()
    Z_true = data[6]
    time = torch.linspace(0, 1.0, len(Z_true)).reshape(-1, 1)
    model = create_model()
    Z_NN = positive_output(model, time)
    z_nn_np = Z_NN.detach().numpy().flatten()
    z_interp = np.interp([0.55, 0.65], time.numpy().flatten(), z_nn_np)
    return z_interp

#function for ode loss
def ode_loss(S_model, E_model, I_model, R_model, beta_model, t, max_time):
    N,_,_,_,_,_ = get_initial_conditions()
    sigma, gamma, omega = get_parama()
    S = positive_output(S_model, t)
    E = positive_output(E_model, t)
    I = positive_output(I_model, t)
    R = positive_output(R_model, t)
    beta = positive_output(beta_model, t)

    #derivative with respect to physical time
    dS_dt = physical_derivative(S, t, max_time)
    dE_dt = physical_derivative(E, t, max_time)
    dI_dt = physical_derivative(I, t, max_time)
    dR_dt = physical_derivative(R, t, max_time)

    #Force of Infection
    FoI = beta * S * I / N

    #SEIR residuals
    r_S = dS_dt + FoI - omega * R
    r_E = dE_dt - FoI + sigma * E
    r_I = dI_dt - sigma * E + gamma * I
    r_R = dR_dt - gamma * I + omega * R

    #ode loss
    loss = torch.mean(r_S**2 + r_E**2 + r_I**2 + r_R**2)

    return loss

#function for data loss
def data_loss(E_model, t, Z_obs_tensor):
    sigma,_, _ = get_parama()
    E = positive_output(E_model, t)
    Z_pred = sigma * E
    loss = torch.mean((Z_pred - Z_obs_tensor)**2)
    return loss

#function for IC loss
def ic_loss(S_model, E_model, I_model, R_model, t0):
    _, S0, E0, I0, R0, _ = get_initial_conditions()
    S = positive_output(S_model, t0)
    E = positive_output(E_model, t0)
    I = positive_output(I_model, t0)
    R = positive_output(R_model, t0)
    loss = torch.mean((S - S0)**2 + (E - E0)**2 + (I - I0)**2 + (R - R0)**2)
    return loss

#structural loss
def struct_loss(S_model, I_model, R_model, beta_model, t, struct_targets):
    s_struct_t, i_struct_t, r_struct_t, beta_struct_t = struct_targets
    S = positive_output(S_model, t)
    I = positive_output(I_model, t)
    R = positive_output(R_model, t)
    beta = positive_output(beta_model, t)
    loss = torch.mean((S - s_struct_t)**2) + torch.mean((I - i_struct_t)**2) + torch.mean((R - r_struct_t)**2) + torch.mean((beta - beta_struct_t)**2)
    return loss

#function for training
def train_model(S_model, E_model, I_model, R_model, beta_model, tau_grad, Z_obs_tensor, struct_targets, weights, max_time, epochs):
    N, S0, E0, I0, R0, _ = get_initial_conditions()
    sigma, gamma, omega = get_parama()
    params = list(S_model.parameters()) + list(E_model.parameters()) + list(I_model.parameters()) + list(R_model.parameters()) + list(beta_model.parameters())
    optimizer = optim.Adam(params, lr= 0.001)
    t0 = tau_grad[0]
    ode_weight, data_weight, ic_weight, struct_weight = weights

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_ode = ode_loss(S_model, E_model, I_model, R_model, beta_model, tau_grad, max_time)
        loss_data = data_loss(E_model, tau_grad, Z_obs_tensor)
        loss_ic = ic_loss(S_model, E_model, I_model, R_model, t0)
        loss_struct = struct_loss(S_model, I_model, R_model, beta_model, tau_grad, struct_targets)

        total_loss = ode_weight * loss_ode + data_weight * loss_data + ic_weight * loss_ic + struct_weight * loss_struct

        total_loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(
                f"total={total_loss.item():.4f}, "
                f"data={loss_data.item():.4f}, "
                f"ode={loss_ode.item():.4f}, "
                f"ic={loss_ic.item():.4f}, "
                f"struct={loss_struct.item():.4f}"
            )

    S_pred = positive_output(S_model, tau_grad)
    E_pred = positive_output(E_model, tau_grad)
    I_pred = positive_output(I_model, tau_grad)
    R_pred = positive_output(R_model, tau_grad)
    beta_pred = positive_output(beta_model, tau_grad)

    return S_pred, E_pred, I_pred, R_pred, beta_pred


#plot true, structural and PINN predictions
def plot_results(t, true_data, struct_data, pinn_data):
    names = ["S", "E", "I", "R", "beta"]
    for j in range(5):
        plt.plot(figsize=(10,5))
        plt.plot(t, true_data[j], label = "true", linewidth = 2)
        plt.plot(t, struct_data[j], label = "structural reconstruction", linestyle = "--")
        plt.plot(t, pinn_data[j], label = "PINN predictions")
        plt.xlabel("Days")
        plt.ylabel(names[j])
        plt.title(f"{names[j]} reconstruction")
        plt.legend()
        plt.savefig(os.path.join(Output_Dir, f"{names[j]}_reconstruction.png"), dpi=200)
        plt.close()

#define the main function
def main():
    #Generate true data and the noisy incidence observations
    (t, S_true, E_true, I_true, R_true, C_true, Z_true,
     beta_fn, sigma, gamma, omega, N, Z_obs) = get_true_data()

    beta_true = beta_fn(t)
    max_time = t[-1]

    tau_data = torch.linspace(0, 1.0, len(t)).reshape(-1, 1)
    Z_obs_tensor = torch.tensor(Z_obs[:, None], dtype=torch.float32)

    # Plot observed incidence
    plt.figure(figsize=(10, 5))
    plt.plot(t, Z_true, label="True incidence")
    plt.scatter(t, Z_obs, s=8, alpha=0.5, label="Noisy incidence")
    plt.xlabel("Days")
    plt.ylabel("Z(t)")
    plt.title("True and observed incidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Output_Dir, "incidence_data.png"), dpi=200)
    plt.close()

    #3. Fit incidence neural network
    incidence_model = fit_incidence_nn(Z_obs_tensor, tau_data)

    #4. Calculate fitted incidence and its derivative
    tau_grad = tau_data.clone().detach().requires_grad_(True)

    z_hat = positive_output(incidence_model, tau_grad)
    dz_hat_dt = physical_derivative(z_hat, tau_grad, max_time)

    z_hat_np = z_hat.detach().numpy().flatten()
    dz_hat_dt_np = dz_hat_dt.detach().numpy().flatten()

    # Structural reconstruction
    _, _, _, I0, R0, _ = get_initial_conditions()

    s_struct, e_struct, i_struct, r_struct, beta_struct = sovle_odes(z_hat_np, dz_hat_dt_np,sigma, gamma, omega, N, I0, R0, t)

     #Convert structural reconstruction to tensors
    s_struct_t = torch.tensor(s_struct[:, None], dtype=torch.float32)

    i_struct_t = torch.tensor(i_struct[:, None], dtype=torch.float32)

    r_struct_t = torch.tensor(r_struct[:, None], dtype=torch.float32)

    beta_struct_t = torch.tensor(beta_struct[:, None], dtype=torch.float32)

    struct_targets = (s_struct_t, i_struct_t, r_struct_t, beta_struct_t)

    #Create five neural networks
    S_model = create_model()
    E_model = create_model()
    I_model = create_model()
    R_model = create_model()
    beta_model = create_model()

    #Train the PINN
    weights = (1.0, 1.0, 1.0, 1000.0)
    epochs = 12000

    S_pred, E_pred, I_pred, R_pred, beta_pred = train_model(S_model, E_model, I_model, R_model, beta_model,
                                                               tau_grad, Z_obs_tensor, struct_targets,weights, max_time, epochs)

    #Convert PINN predictions to NumPy
    S_pred = S_pred.detach().numpy().flatten()
    E_pred = E_pred.detach().numpy().flatten()
    I_pred = I_pred.detach().numpy().flatten()
    R_pred = R_pred.detach().numpy().flatten()
    beta_pred = beta_pred.detach().numpy().flatten()

    #prepare data for plotting
    true_data = (S_true, E_true, I_true, R_true, beta_true)

    struct_data = (s_struct, e_struct, i_struct, r_struct, beta_struct)

    pinn_data = (S_pred, E_pred, I_pred, R_pred, beta_pred)

    # Plot all compartments and beta
    plot_results(t, true_data, struct_data, pinn_data)

    # Save numerical predictions
    predictions = pd.DataFrame({
        "t": t,
        "S_true": S_true,
        "S_struct": s_struct,
        "S_PINN": S_pred,
        "E_true": E_true,
        "E_struct": e_struct,
        "E_PINN": E_pred,
        "I_true": I_true,
        "I_struct": i_struct,
        "I_PINN": I_pred,
        "R_true": R_true,
        "R_struct": r_struct,
        "R_PINN": R_pred,
        "beta_true": beta_true,
        "beta_struct": beta_struct,
        "beta_PINN": beta_pred
    })

    predictions.to_csv(os.path.join(Output_Dir, "predictions.csv"),index=False)

if __name__ == "__main__":
    main()
       















