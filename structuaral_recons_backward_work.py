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
    return t, S, E, I, R, C, Z, beta_fn, sigma, gamma, omega, N

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

def physical_derivative(y, tau, max_time):
    return derivative(y, tau) / max_time

#fitting the NN to the daily incidence data
def fit_incidence_nn(z_obs_tensor, tau_data):
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for epoch in range(5000):
        optimizer.zero_grad()
        z_hat = positive_output(model, tau_data)
        loss = torch.mean((z_hat - z_obs_tensor)**2)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

#load saved incidence model
def load_saved_incidence_model():
    path = os.path.join(Output_Dir, "incidence_model.pt")
    model = create_model().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

#The reconstruction equation integration part

def integrate_linear_ode(dydt, y0, t_span, t_eval):
    def ode_system(t, y):
        return dydt(t, y)
    sol = solve_ivp(ode_system, t_span, [y0], method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9)
    return sol.y[0]

#reconstruction from incidence
def reconstruction_from_incidence(z_hat, beta_fn, sigma, gamma, omega, N, S0, E0,
                                  I0, R0, C0, t):
    e = z_hat / sigma
    # np.interp is used to interpolate the incidence data for integration
    i = integrate_linear_ode(lambda tt, I: np.interp(tt, t, z_hat) - gamma * I, I0, (0, len(t) - 1), t)
    r = integrate_linear_ode(lambda tt, R: gamma * np.interp(tt, t, i) - omega * R, R0, (0, len(t) - 1), t)
    s = N - e - i - r
    return s, e, i, r

#Main function to run
def main():
    (t, S_true, E_true, I_true, R_true, C_true, Z_true, beta_true,
    sigma, gamma, omega, N) = get_true_data()
    max_time = t[-1]
    tau_data = torch.linspace(0, 1.0, len(t), device = device).reshape(-1, 1)
    z_obs = torch.tensor(Z_true[:, None], device = device)

    if Train:
        nn_model = fit_incidence_nn(z_obs, tau_data)
        torch.save(nn_model.state_dict(), os.path.join(Output_Dir,
                                                       "incidence_model.pt"))
    else:
        nn_model =load_saved_incidence_model()

    tau_grad = tau_data.clone().requires_grad_(True)
    z_hat = positive_output(nn_model, tau_grad)
    dz_hat_dt = physical_derivative(z_hat, tau_grad, max_time)
    z_hat_np = z_hat.detach().cpu().numpy().flatten()
    dz_hat_dt_np = dz_hat_dt.detach().cpu().numpy().flatten()

    rec_nn = reconstruction_from_incidence(z_hat_np, beta_true, sigma, gamma, omega, N,
                                           S_true[0], E_true[0], I_true[0],
                                           R_true[0], C_true[0], t)

    pd.DataFrame({
        "Days": t,
        "S_true": S_true, "S_rec_after_NN": rec_nn[0],
        "E_true": E_true, "E_rec_after_NN": rec_nn[1],
        "I_true": I_true, "I_rec_after_NN": rec_nn[2],
        "R_true": R_true, "R_rec_after_NN": rec_nn[3],
        "C_true": C_true,
        "Z_true": Z_true, "Z_NN": z_hat_np, "dZ_NN_dt": dz_hat_dt_np

    }).to_csv(os.path.join(Output_Dir, "reconstruction_results.csv"), index= False)

    #plots
    plt.figure(figsize=(10, 5))
    plt.plot(t, Z_true, label = "True Incidence", linewidth = 1.5)
    plt.plot(t, z_hat_np, label = "NN-smoothed Incidence", linewidth = 1)
    plt.xlabel("Days")
    plt.ylabel("Incidence")
    plt.title("Incidence Reconstruction")
    plt.legend()
    plt.savefig(os.path.join(Output_Dir, "incidence_reconstruction.png"))
    plt.close()

    for name, true_arr, rec_arr in [
        ("S", S_true, rec_nn[0]),
        ("E", E_true, rec_nn[1]),
        ("I", I_true, rec_nn[2]),
        ("R", R_true, rec_nn[3])
    ]:
        plt.figure(figsize=(10, 5))
        plt.plot(t, true_arr, label = f"True {name}", linewidth = 1.5)
        plt.plot(t, rec_arr, label = f"Reconstructed {name}", linewidth = 1)
        plt.xlabel("Days")
        plt.ylabel(name)
        plt.title(f"{name} Reconstruction")
        plt.legend()
        plt.savefig(os.path.join(Output_Dir, f"{name}_reconstruction.png"))
        plt.close()

if __name__ == "__main__":
    main()



    



