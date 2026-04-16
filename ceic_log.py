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
from scipy.stats import gamma as gamma_dist

#%%
#Canada Data
def get_canada_data():
    df = pd.read_csv('cases_can.csv')  
    df['date'] = pd.to_datetime(df['date'])
    inc_cases = df['value_daily'].rolling(window=7).mean().dropna()
    return inc_cases

#%%
#Variant periods in Canada based on public health data and reports, with start and end dates for each variant's dominance
CANADA_VARIANTS = {
    'Alpha':   (datetime.datetime(2020, 12, 27), datetime.datetime(2021, 6, 30)),
    'Delta':   (datetime.datetime(2021, 7, 1),   datetime.datetime(2021, 12, 14)),
    'Omicron': (datetime.datetime(2021, 12, 15), datetime.datetime(2022, 5, 31)),
    'BA.5':    (datetime.datetime(2022, 6, 1),   datetime.datetime(2023, 1, 31)),
}

#%%
#Variant parameters (sigma and gamma) based on literature estimates of incubation and infectious periods for each variant, converted to rates (1/days)
VARIANT_PARAMS = {
    'Alpha': (1/5.2, 1/10),  # sigma, gamma
    'Delta': (1/4.5, 1/8),
    'Omicron': (1/3.5, 1/6),
    'BA.5': (1/3.0, 1/5)
}

#%%

def get_parama(total_days):
    #sigma and gamma parameters for the variants in covid
    start_date = datetime.datetime(2020, 1, 23)

    #create empty arrays for sigma and gamma 
    sigma_t = np.zeros(total_days)
    gamma_t = np.zeros(total_days)

    #loop through each day and assign sigma and gamma based on the variant period
    for i in range(total_days):
        current_date = start_date + datetime.timedelta(days=i)
        for variant, (start, end) in CANADA_VARIANTS.items():
            if start <= current_date <= end:
                sigma_t[i], gamma_t[i] = VARIANT_PARAMS[variant]
                break
        else:
            sigma_t[i], gamma_t[i] = VARIANT_PARAMS['Alpha'] # default to Alpha parameters if no variant matches

    params = sigma_t, gamma_t
    return params

#%%
#PINN model
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 6)
    )
    return model

#%%
#Apply constraints to get positive values
def forward_with_constraints(model, t):
    raw = model(t)
    S = torch.sigmoid(raw[:, 0:1])
    E = F.softplus(raw[:, 1:2])
    I = F.softplus(raw[:, 2:3])
    R = F.softplus(raw[:, 3:4])
    C_inci = F.softplus(raw[:, 4:5])
    beta = F.softplus(raw[:, 5:6])
    return torch.cat([S, E, I, R, C_inci, beta], dim=1)

#%%
#ODE loss
def ode_loss(model, t, params, T_max):
    sigma_array, gamma_array = params
    out = forward_with_constraints(model, t)
   
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    R = out[:, 3]
    C_inci = out[:, 4]
    beta = out[:, 5]

   #creates a tensor of all 1's same shape as S, need this to combine gradients from multiple outputs
    ones = torch.ones_like(S)
    #create_graph= True, because we need derivative of a derivative otherwise the term would consider as a constant
   
    dSdt = grad.grad(S, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(E, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIdt = grad.grad(I, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dRdt = grad.grad(R, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dC_incidt = grad.grad(C_inci, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dbetadt = grad.grad(beta, t, grad_outputs=ones, create_graph=True)[0].squeeze()
   
    #convert sigma and gamma arrays to tensors and get the values at the corresponding time points
    sigma = torch.tensor(sigma_array, dtype=torch.float32)
    gamma = torch.tensor(gamma_array, dtype=torch.float32)
    Trans = beta * S * I # beta * S * I

    r_S = dSdt + T_max * Trans
    r_E = dEdt - T_max * (Trans - sigma * E)
    r_I = dIdt - T_max * (sigma * E - gamma * I)
    r_R = dRdt - T_max * (gamma * I)
    r_C_inci = dC_incidt - T_max * (sigma * E)

    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2 + r_C_inci**2).mean()
    return loss



#%%
#Initial condition loss
def ic_loss(model, ICs, T_max, params):
    sigma_array, gamma_array = params
    S0, E0, I0, R0, C_inci0 = ICs
    
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    out = forward_with_constraints(model, t0)
    
    # Value loss - keep gradients!
    pred = out[0, : 5]  # S, E, I, R, C_inci at t =0
    target = torch.tensor([S0, E0, I0, R0, C_inci0], dtype=torch.float32)
    v_loss =  ((pred - target)**2).mean()
   
    # Derivative loss
    ones = torch.ones(1)
    dSdt = grad.grad(out[:,0], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(out[:,1], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIdt = grad.grad(out[:,2], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dRdt = grad.grad(out[:,3], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    dC_incidt = grad.grad(out[:,4], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    
    #get sigma and gamma at t=0
    sigma = torch.tensor(sigma_array[0], dtype=torch.float32)
    gamma = torch.tensor(gamma_array[0], dtype=torch.float32)
    Trans = out[0,5] * out[0,0] * out[0,2] # beta * S * I at t=0
   
    r_S = dSdt +  T_max * Trans
    r_E = dEdt -  T_max * (Trans - sigma * out[0,1])
    r_I = dIdt - T_max * (sigma * out[0,1] + gamma * out[0,2])
    r_R = dRdt - T_max * (gamma * out[0,2])
    r_C_inci = dC_incidt - T_max * (sigma * out[0,1])
   
    d_loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2 + r_C_inci**2).mean()
   
    return v_loss + d_loss

#%%
# data loss
def data_loss(model, t_np, I_obs):
   
    indices = np.arange(0, len(t_np), 7)
    
    t_k = torch.tensor(t_np[indices].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    cum_obs = torch.tensor(np.cumsum(I_obs)[indices], dtype=torch.float32)
   
    out = forward_with_constraints(model, t_k)
    C_inci = out[:, 4]
    
    # v loss (cumulative)
    v_loss = ((C_inci - cum_obs) ** 2).mean()
    
    return v_loss

#%%
#Time to train the model
def time_to_train():
    cases = get_canada_data()
    total_days = len(cases)
    t_np_full = np.linspace(0, 1, total_days, dtype=np.float32)
    t_full = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    return t_np_full, t_full, total_days, cases
#%%
#Train model
def train_model(epochs=40000, test_days=100, save=False):
    # general parameters used in the model
    N = 38000000
    ICs = [(N-2)/N, 0/N, 0.0, 2/N, 0.0] # no covid cases at the start, seed with 1 or 2 or 10
    t_np_full, t_full, total_days, cases = time_to_train()
    Ir_obs = cases.values / N # convert cases to a proportion
    
    # Split into train and test
    train_days = total_days - test_days
    train_indices = np.arange(0, train_days)

    # Use TOTAL days for T_max
    T_max = float(total_days)
    
    # Training time tensor (only training points)
    t_train_np = t_np_full[train_indices] # training time
    t_train_tensor = torch.tensor(t_train_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True) 
    Ir_train = Ir_obs[train_indices]      # and cases at those time points
    
    #time_varying parameters for the variants
    params_full = get_parama(total_days)
    sigma_array, gamma_array = params_full
    params = sigma_array[train_indices], gamma_array[train_indices] # only training period parameters

    # Create model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #weights for different loss compartments
    def get_weights(ep):
    # gradually increase importance of data
      w_ic = 100 + 0.02 * ep
      w_ode = 1 + 0.001 * ep
      w_data = 10 + 0.05 * ep
    
      return w_ic, w_ode, w_data

    print(f"{'Ep':>6} {'IC':>12} {'ODE':>12} {'CEIC':>12} {'Total':>12}  S_range       beta_range")
   
    history = []
    for ep in range(epochs):
        w_ic , w_ode, w_data = get_weights(ep)
        optimizer.zero_grad() #reset any previous gradients
       
        # IC loss (at t=0)
        l_ic = ic_loss(model, ICs, T_max, params)
       
        # ODE loss - Using FULL time tensor
        l_ode = ode_loss(model, t_train_tensor, params, T_max)
       
        # Data loss - Only on training points
        l_data = data_loss(model, t_train_np, Ir_train)
       
        # Combined loss - Adjusted weights
        loss = w_ic * l_ic + w_ode * l_ode + w_data * l_data 
        loss.backward()
        
        # Gradient clipping
        #calculate L2 norm for all gradients, if its >1 then scale it to norm=1, and if <1 then does nothing
        #we need this to be safe from exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        history.append(loss.item()) #loss is a tensor with attached graph, it remembers how it was computed,
        #takes lot of memory, we need to convert it to a scalar value so 
        # we detach it from the graph and then convert to a python number using item()
       
        if ep % 1000 == 0:
            with torch.no_grad():
                out = forward_with_constraints(model, t_full)
                s_range = f"[{out[:,0].min():.4f}, {out[:,0].max():.4f}]"
                b_range = f"[{out[:,5].min():.3f}, {out[:,5].max():.3f}]"
            print(f"{ep:} {l_ic.item():} {l_ode.item():} {l_data.item():} "
                  f"{loss.item():}  {s_range}  {b_range}")
    if save: 
        torch.save(model.state_dict(), "pinn_model.pth") 
    return model, history, train_days

#%%
#Plot
def plot_all(model, train_days,t_full):
    t_np_full, t_full, total_days, cases = time_to_train()
    days = np.arange(total_days)
   
    # Convert cases to numpy
    cases_np = np.array(cases)
    Ic_obs   = np.cumsum(cases_np)
   
    # Model prediction
    with torch.no_grad():
        out = forward_with_constraints(model, t_full).numpy()
   
    S = out[:, 0] * N
    E = out[:, 1] * N
    I = out[:, 2] * N
    R = out[:, 3] * N
    C_inci = out[:, 4] * N
    beta = out[:, 5]
   
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
    plt.plot(days, C_inci, 'b-', label='PINN Prediction (Cr)', linewidth=2)
    plt.plot(days[:train_days], Ic_obs[:train_days], 'g.', label='Training Data', markersize=3)
    plt.plot(days[train_days:], Ic_obs[train_days:], 'r.', label='Test Data', markersize=3)
    plt.axvline(train_days, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Reported Cases')
    plt.title('PINN Cr vs Cumulative Observed Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("reported_cumulative_cases_split.png")
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
    plt.plot(days, C_inci/1e3)
    plt.title(f'Cumulative Incidence (final: {C_inci[-1]/1e3:.1f}k)')
    plt.xlabel('Days')
    plt.ylabel('cumulative cases (thousands)')
    plt.grid(True)
    plt.savefig("all_plots.png")
    plt.close()

    # Validation Metrics
    #test days starts after the train days
    pred_test = I[train_days:]
    true_test = cases_np[train_days:]
   
    mse = np.mean((pred_test - true_test)**2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(pred_test, true_test)[0, 1]
    mae = np.mean(np.abs(pred_test - true_test))
    print(f"RMSE: {rmse:.0f} cases")
    print(f"MAE:  {mae:.0f} cases")
    print(f"MSE:  {mse:.4e}")
    print(f"R:    {corr:.4f}")
    print(f"Beta range:    [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"Total recovered: {R[-1]/1e6:.3f}M")

def load_model(path):
    model = create_model()
    model.load_state_dict(torch.load(path))
    return model

#%%
#Run everything
if __name__ == "__main__":
    
    #train_model(epochs=50000, test_days=100, save=True)
    # run_training(save=True)
    model = load_model("pinn_model.pth")
    t_np_full, t_full, total_days, cases = time_to_train()

    with torch.no_grad():
        out = forward_with_constraints(model, t_full).numpy()
        N = 38000000
    
    print(out.shape)
    print(out[:, 4].shape)
    plt.plot(out[:,4]*N )
    plt.plot(np.cumsum(cases))
    plt.savefig("cumulative_cases.png")
    plt.close()

    plt.plot(np.diff(out[:,4]*N ))
    plt.plot(cases)
    plt.savefig("daily_cases.png")
    plt.close()

    plot_all(model, train_days=total_days-100, t_full=t_full)
    

def R_eff(out, params):
    sigma, gamma = params
    S    = out[:, 0]   # proportion, no need to multiply by N
    beta = out[:, 5]
    return (beta * S) / gamma   # dimensionless

#Cori et al. (2013) framework used in EpiEstim, 
# with a discretized Gamma serial interval distribution, 
# and a Bayesian posterior for R_t given Poisson incidence data. 
# This is a more traditional epidemiological estimate of R_t that we can compare to the PINN-derived R_t.
def estimate_R_t(cases, mean_si=5.2, sd_si=1.5, window=3, a0=1, b0=2):
    """True EpiEstim Bayesian R_t: posterior Gamma given Poisson incidence.
        mean_si, sd_si - serial interval parameters for the discretized serial interval distribution (Gamma)
        window- assume R_t is constant over this many days, and use that window 
        to calculate the likelihood of observed cases given the past infectiousness (Lambda) 
        and the prior (a0, b0) to get the posterior distribution of R_t at each time point.
    """
    cases = np.array(cases, dtype=float)
    T = len(cases)

    #serial interval distribution
    shape_si = (mean_si / sd_si) ** 2
    scale_si = (sd_si ** 2) / mean_si
    s = np.arange(1, 15)  # serial interval up to 15 days
    w = gamma_dist.pdf(s, a=shape_si, scale=scale_si)
    w = w / w.sum()  # normalize

    #compute pointwise lambda (infectiousness) as the convolution of past cases with the serial interval distribution
    lambda_t = np.zeros(T)
    for t in range (len(w), T):
        # lambda_t[t] is the sum of past cases weighted by the serial interval distribution, which gives the expected number of new infections generated by the past cases at time t
        # lambda_t = ∑ (I[t-s] * w[s]) for s=1 to max serial interval, where I[t-s] is the number of cases at time t-s and w[s] is the probability that a case infected at time t-s would generate a new case at time t based on the serial interval distribution
        lambda_t[t] = sum(cases[t - s] * w[s - 1] for s in range(1, len(w)+1))  # past cases weighted by serial interval
    #calculate the posterior distribution of R_t at each time point using the likelihood of observed cases given lambda and the prior
    Rt_mean = np.zeros(T)
    Rt_lower = np.zeros(T)
    Rt_upper = np.zeros(T)
    
    #sliding window approach to estimate R_t, assuming it is constant over the window
    for t in range(window + len(w), T):

        #sum of cases in the window, and sum of lambda in the window
        cases_window = cases[t - window + 1:t + 1]
        lambda_window = lambda_t[t - window + 1:t + 1]
        sum_cases = cases_window.sum()
        sum_lambda = lambda_window.sum()
        #posterior parameters for R_t given the likelihood of observed cases and the prior
        a_post = a0 + sum_cases
        #posterior mean and 95% credible interval for R_t
        scale_post = 1 / (1 / b0 + sum_lambda)
        Rt_mean[t] = a_post * scale_post
        # ppf = Percent Point Function (inverse of CDF) which gives the value below which a given percentage of data falls, used here to get the 2.5th and 97.5th percentiles for the credible interval of R_t
        Rt_lower[t] = gamma_dist.ppf(0.025, a=a_post, scale=scale_post)
        Rt_upper[t] = gamma_dist.ppf(0.975, a=a_post, scale=scale_post)
    #set early R_t estimates to NaN since they are unreliable due to lack of data
    Rt_mean[:window + len(w)] = np.nan
    Rt_lower[:window + len(w)] = np.nan
    Rt_upper[:window + len(w)] = np.nan
    return Rt_mean, Rt_lower, Rt_upper

#%%
#Plot beta and R_t with variant periods shaded, using both the PINN-derived R_t 
# and the Bayesian R_t estimates from EpiEstim for comparison. 
# This will allow us to see how the transmission rate and effective reproduction number 
# evolved over time, and how they correspond to the different variant periods in Canada.
def plot_beta_R_t(out, cases, params, variants, data_start_date):
    cases_np = np.array(cases)
    days = np.arange(len(cases_np))
    
    beta = out[:, 5]
    Rt_pinn = R_eff(out, params)
    Rt_mean, Rt_lower, Rt_upper = estimate_R_t(cases_np)

    # Correct day offset using actual data start date
    def date_to_day(dt):
        return (dt - data_start_date).days

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    variant_colors = ['purple', 'orange', 'green', 'brown']

    # Beta panel
    axes[0].plot(days, beta, color='blue', linewidth=2)
    axes[0].set_ylabel('β(t)')
    axes[0].set_title('PINN-Learned Transmission Rate β(t)')
    axes[0].grid(True)

    # R_t panel
    valid = ~np.isnan(Rt_lower) & ~np.isnan(Rt_upper)
    axes[1].plot(days, Rt_pinn,  color='orange', linewidth=2, label='R_t (PINN)')
    axes[1].plot(days[valid], Rt_mean[valid],  color='green',  linewidth=2, label='R_t (Bayesian)')

    # Confidence interval shading for Bayesian R_t that handles NaNs
    axes[1].fill_between(days[valid], Rt_lower[valid], Rt_upper[valid], color='grey', alpha=0.3, label='95% CI (Bayesian)')
    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='R_t = 1')
    axes[1].set_ylabel('R_t')
    axes[1].set_title('Effective Reproduction Number with Variant Periods')
    axes[1].set_ylim(0, 5)
    axes[1].grid(True)

    #lower limit and upper limit of x-axis based on the data
    axes[1].set_xlim(0, len(cases_np))


    # Shade variants on both panels
    for (variant, (start, end)), col in zip(variants.items(), variant_colors):
        s_day = date_to_day(start)
        e_day = date_to_day(end)

        for ax in axes:
            ax.axvspan(s_day, e_day, alpha=0.15, color=col, label=variant)

    axes[1].legend(loc='upper right')
    plt.xlabel('Days from data start')
    plt.tight_layout()
    plt.savefig("beta_Rt_variants.png")
    plt.close()

#parameters
params =  get_parama(total_days)

plot_beta_R_t(out, cases, params, CANADA_VARIANTS, data_start_date=datetime.datetime(2020, 1, 21)) 
#print valid R_t range during each variant period
for variant, (start, end) in CANADA_VARIANTS.items():
    s_day = (start - datetime.datetime(2020, 1, 23)).days
    e_day = (end - datetime.datetime(2020, 1, 23)).days

    #slice both out and params to the variant period, and calculate R_t range for that period
    out_slice = out[s_day:e_day]
    sigma_slice , gamma_slice = params[0][s_day:e_day], params[1][s_day:e_day]
    params_slice = sigma_slice, gamma_slice
    Rt_variant = R_eff(out_slice, params_slice)
    print(f"{variant}: R_t range = [{Rt_variant.min():.2f}, {Rt_variant.max():.2f}]")


Rt_mean, Rt_lower, Rt_upper = estimate_R_t(cases, mean_si=5.2, sd_si=1.5, window=3, a0=1.0, b0=5.0)

