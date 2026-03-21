#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as grad
import torch.nn.functional as F

#%%
#Canada Data
def get_canada_data():
    df = pd.read_csv('/Users/srejon/Downloads/cases_can.csv')
    df['date'] = pd.to_datetime(df['date'])
    inc_cases = df['value_daily'].rolling(window=7).mean().dropna()
    return inc_cases

#%%
#PINN model
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 8)
    )
    return model

#%%
#Apply constraints to get positive values
def forward_with_constraints(model, t):
    raw = model(t)
    S = torch.sigmoid(raw[:, 0:1])
    E = F.softplus(raw[:, 1:2])
    Iu = F.softplus(raw[:, 2:3])
    Ir = F.softplus(raw[:, 3:4])
    H = F.softplus(raw[:, 4:5])
    R = F.softplus(raw[:, 5:6])
    D = F.softplus(raw[:, 6:7])
    beta = F.softplus(raw[:, 7:8]) + 0.01
    return torch.cat([S, E, Iu, Ir, H, R, D, beta], dim=1)

#%%
#ODE loss
# derivatives multiplied by T_max because we normalized t [0,1], because this stabilize the training)
#  By chain rule: d(state)/dt_normalised = T_max * d(state)/dt_real
def ode_loss(model, t, params, T_max):
    sigma, gamma_u, gamma_r, p, h, gamma_h, mu = params
    out = forward_with_constraints(model, t)
   
    S = out[:, 0]
    E = out[:, 1]
    Iu = out[:, 2]
    Ir = out[:, 3]
    H = out[:, 4]
    R = out[:, 5]
    D = out[:, 6]
    beta = out[:, 7]
   
    ones = torch.ones_like(S)
   
    dSdt = grad.grad(S, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(E, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIudt = grad.grad(Iu, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIrdt = grad.grad(Ir, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dHdt = grad.grad(H, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dRdt = grad.grad(R, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dDdt = grad.grad(D, t, grad_outputs=ones, create_graph=True)[0].squeeze()
   
    
    Trans = beta * S * (Iu + Ir)
   
    r_S = dSdt + T_max * Trans
    r_E = dEdt - T_max * (Trans - sigma * E)
    r_Iu = dIudt - T_max * ((1-p) * sigma * E - gamma_u * Iu)
    r_Ir = dIrdt - T_max * (p * sigma * E - (gamma_r + h) * Ir)
    r_H = dHdt - T_max * (h * Ir - (gamma_h + mu) * H)
    r_R = dRdt - T_max * (gamma_u*Iu + gamma_r*Ir + gamma_h*H)
    r_D = dDdt - T_max * (mu * H)
   
    return (r_S**2 + r_E**2 + r_Iu**2 + r_Ir**2 + r_H**2 + r_R**2 + r_D**2).mean()

#%%
#Initial condition loss
#IC LOSS WITH CEIC AT t=0
# Standard IC loss matches values at t=0.
# CEIC additionally matches the DERIVATIVE at t=0 to the ODE RHS.
# This satisfies both position AND slope at the start, preventing early divergence.
def ic_loss(model, ICs, T_max, params):
    sigma, gamma_u, gamma_r, p, h, gamma_h, mu = params
    S0, E0, Iu0, Ir0, H0, R0, D0 = ICs
   
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    out = forward_with_constraints(model, t0)
   
    # Value loss
    pred = torch.tensor([out[0,0].detach().item(), out[0,1].detach().item(),
                         out[0,2].detach().item(), out[0,3].detach().item(),
                         out[0,4].detach().item(), out[0,5].detach().item(),
                         out[0,6].detach().item()], dtype=torch.float32)
    target = torch.tensor([S0, E0, Iu0, Ir0, H0, R0, D0], dtype=torch.float32)
    v_loss = 500* ((pred - target)**2).mean()
   
    # Derivative loss
    ones = torch.ones(1)
    dSdt = grad.grad(out[:,0], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(out[:,1], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIudt = grad.grad(out[:,2], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIrdt = grad.grad(out[:,3], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dHdt = grad.grad(out[:,4], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dRdt = grad.grad(out[:,5], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dDdt = grad.grad(out[:,6], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
   
    Trans = out[0,7] * out[0,0] * (out[0,2] + out[0,3])
   
    r_S = dSdt + T_max * Trans
    r_E = dEdt - T_max * (Trans - sigma * out[0,1])
    r_Iu = dIudt - T_max * ((1-p) * sigma * out[0,1] - gamma_u * out[0,2])
    r_Ir = dIrdt - T_max * (p * sigma * out[0,1] - (gamma_r + h) * out[0,3])
    r_H = dHdt - T_max * (h * out[0,3] - (gamma_h + mu) * out[0,4])
    r_R = dRdt - T_max * (gamma_u*out[0,2] + gamma_r*out[0,3] + gamma_h*out[0,4])
    r_D = dDdt - T_max * (mu * out[0,4])
   
    d_loss = (r_S**2 + r_E**2 + r_Iu**2 + r_Ir**2 + r_H**2 + r_R**2 + r_D**2).mean()
   
    return v_loss + d_loss

#%%
#CEIC data loss
def ceic_data_loss(model, t_np, Ir_obs_np, params, T_max):
    sigma, gamma_u, gamma_r, p, h, gamma_h, mu = params
   
    # Use every 7th point(not average)
    indices = np.arange(0, len(t_np), 7)
    t_k = torch.tensor(t_np[indices].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    Ir_k = torch.tensor(Ir_obs_np[indices], dtype=torch.float32)
   
    out = forward_with_constraints(model, t_k)
    S = out[:, 0]
    E = out[:, 1]
    Iu = out[:, 2]
    Ir = out[:, 3]
    beta = out[:, 7]
   
    # Value loss
    v_loss = ((Ir - Ir_k)**2).mean()
   
    # Derivative losses
    #compute the derivative at each point, give equal weight and return the vector of derivatives d/dt
    ones = torch.ones(len(indices))
    dIrdt = grad.grad(Ir, t_k, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(E, t_k, grad_outputs=ones, create_graph=True)[0].squeeze()
   
    rhs_Ir = T_max * (p * sigma * E - (gamma_r + h) * Ir_k)
   
    Trans = beta * S * (Iu + Ir_k)
    #E consistency — the beta-identifying constraint
    rhs_E = T_max * (Trans - sigma * E)
   
    d_loss = ((dIrdt - rhs_Ir)**2).mean() + ((dEdt - rhs_E)**2).mean()
   
    return v_loss + d_loss

#%%
#Train model
def train_model(epochs=40000, test_days=100):
    N = 38000000
    params = (1/5.2, 1/14, 1/14, 0.5, 0.05, 1/21, 0.01)
    ICs = [(N-7)/N, 5/N, 0.0, 2/N, 0.0, 0.0, 0.0]
   
    cases = get_canada_data()
    total_days = len(cases)
    Ir_obs = cases.values / N
   
    # Split into train and test
    train_days = total_days - test_days
    train_indices = np.arange(0, train_days)
    test_indices = np.arange(train_days, total_days)

    # Use TOTAL days for T_max
    T_max = float(total_days)
   
    # Full time tensor for evaluation
    t_np_full = np.linspace(0, 1, total_days, dtype=np.float32)
    t_full = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
   
    # Training time tensor (only training points)
    t_train_np = t_np_full[train_indices]
    t_train = torch.tensor(t_train_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    Ir_train = Ir_obs[train_indices]
   
    # Create model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
   
    history = []
    for ep in range(epochs):
        optimizer.zero_grad()
       
        # IC loss (at t=0)
        l_ic = ic_loss(model, ICs, T_max, params)
       
        # ODE loss - Using FULL time tensor
        l_ode = ode_loss(model, t_full, params, T_max)
       
        # Data loss - Only on training points
        #l_data = data_loss(model, t_train_np, Ir_train, params, T_max)
        l_data = ceic_data_loss(model, t_train_np, Ir_train, params, T_max)
       
        # Combined loss - Adjusted weights
        loss = 3500.0 * l_ic + 1.0 * l_ode + 500.0 * l_data
        loss.backward()
        #gradient clipping, making sure not too big gradient otherwise
        #gradient will explode. If the norm exceeds 1.0, it scales all gradients down so the norm becomes exactly 1.0
        #and if norm less thena 1 does nothing
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        history.append(loss.item())
       
        if ep % 1000 == 0:
            with torch.no_grad():
                out = forward_with_constraints(model, t_full)
                s_range = f"[{out[:,0].min():.4f}, {out[:,0].max():.4f}]"
                b_range = f"[{out[:,7].min():.3f}, {out[:,7].max():.3f}]"
            print(f"{ep:} {l_ic.item():} {l_ode.item():} {l_data.item():} "
                  f"{loss.item():}  {s_range}  {b_range}")
   
    return model, history, cases, N, t_full, train_days

#%%
#Plot
def plot_all(model, history, cases, N, t, train_days):
    total_days = len(cases)
    days = np.arange(total_days)
   
    # Convert cases to numpy
    cases_np = np.array(cases)
   
    # Model prediction
    with torch.no_grad():
        out = forward_with_constraints(model, t).numpy()
   
    S = out[:, 0] * N
    E = out[:, 1] * N
    Iu = out[:, 2] * N
    Ir = out[:, 3] * N
    H = out[:, 4] * N
    R = out[:, 5] * N
    D = out[:, 6] * N
    beta = out[:, 7]
   
    # 1. Training History
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True)
    plt.show()
   
    # 2. Reported Cases (Train/Test split)
    plt.figure(figsize=(12, 5))
    plt.plot(days, Ir, 'b-', label='PINN Prediction', linewidth=2)
    plt.plot(days[:train_days], cases_np[:train_days], 'g.', label='Training Data', markersize=3)
    plt.plot(days[train_days:], cases_np[train_days:], 'r.', label='Test Data', markersize=3)
    plt.axvline(train_days, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    plt.xlabel('Days')
    plt.ylabel('Reported Cases')
    plt.title('PINN vs Observed Data')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    # 3. All Compartments
    plt.figure(figsize=(14, 8))
   
    plt.subplot(2, 4, 1)
    plt.plot(days, S/1e6)
    plt.title(f'Susceptible (final: {S[-1]/1e6:.1f}M)')
    plt.xlabel('Days')
    plt.ylabel('Millions')
    plt.grid(True)
   
    plt.subplot(2, 4, 2)
    plt.plot(days, E/1e3)
    plt.title(f'Exposed (peak: {E.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.grid(True)
   
    plt.subplot(2, 4, 3)
    plt.plot(days, Iu/1e3)
    plt.title(f'Unreported (peak: {Iu.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.grid(True)
   
    plt.subplot(2, 4, 4)
    plt.plot(days, Ir/1e3, 'b-', label='PINN')
    plt.plot(days, cases_np/1e3, 'k--', alpha=0.5, label='Observed')
    plt.title(f'Reported (peak: {Ir.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(2, 4, 5)
    plt.plot(days, H/1e3)
    plt.title(f'Hospitalized (peak: {H.max()/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.grid(True)
   
    plt.subplot(2, 4, 6)
    plt.plot(days, R/1e6)
    plt.title(f'Recovered (final: {R[-1]/1e6:.1f}M)')
    plt.xlabel('Days')
    plt.ylabel('Millions')
    plt.grid(True)
   
    plt.subplot(2, 4, 7)
    plt.plot(days, D/1e3)
    plt.title(f'Deaths (final: {D[-1]/1e3:.0f}k)')
    plt.xlabel('Days')
    plt.ylabel('Thousands')
    plt.grid(True)
   
    plt.subplot(2, 4, 8)
    plt.plot(days, beta)
    plt.title(f'Beta (range: {beta.min():.2f}-{beta.max():.2f})')
    plt.xlabel('Days')
    plt.ylabel('Beta')
    plt.grid(True)
   
    plt.tight_layout()
    plt.show()
   
   
    # 5. Validation Metrics
    pred_test = Ir[train_days:]
    true_test = cases_np[train_days:]
   
    mse = np.mean((pred_test - true_test)**2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(pred_test, true_test)[0, 1]
    mae = np.mean(np.abs(pred_test - true_test))
    print(f"RMSE: {rmse:.0f} cases")
    print(f"MAE:  {mae:.0f} cases")
    print(f"MSE:  {mse:.4e}")
    print(f"R:    {corr:.4f}")
    print(f"% infected:    {(S[0] - S[-1])/S[0]*100:.1f}%")
    print(f"Beta range:    [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"Ir peak:       {Ir.max()/1e3:.1f}k  (observed: {cases_np.max()/1e3:.1f}k)")
    print(f"Total dead:    {D[-1]/1e3:.1f}k")
    print(f"Total recovered: {R[-1]/1e6:.3f}M")

#%%
#Run everything
if __name__ == "__main__":
    model, history, cases, N, t, train_days = train_model(epochs=2000, test_days=100)
    plot_all(model, history, cases, N, t, train_days)



   
  
   
   



