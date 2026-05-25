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
from scipy.stats import ttest_rel
import data_gen

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

data = data_gen.seasonal_beta(beta0 = 0.3, A= 0.2, T =180, phase = 0)(np.linspace(0,365,365))

def get_canada_data():
    df = data_gen.get_inci_data(type= 'seasonal')
    return df

#import data_gen

# def get_canada_data():
#     df = pd.read_csv('us_counties_covid19_daily.csv')
#     df['date'] = pd.to_datetime(df['date'])
#     #choose a specific county, for example, Los Angeles County in California
#     county_df = df[df['county'] == 'Los Angeles']
#     inc_cases = county_df['cases'].rolling(window=7).mean().dropna()
#     return inc_cases



#%%
# #Canada Data
# def get_canada_data():
#     df = pd.read_csv('cases_can.csv')  
#     df['date'] = pd.to_datetime(df['date'])
#     inc_cases = df['value_daily'].rolling(window=7).mean().dropna()
#     return inc_cases

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
def ode_loss(model, t, params, T_max, epsilon, causal = False):
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
    #force of infection term
    Trans = beta * S * I # beta * S * I

    r_S = dSdt + T_max * Trans
    r_E = dEdt - T_max * (Trans - sigma * E)
    r_I = dIdt - T_max * (sigma * E - gamma * I)
    r_R = dRdt - T_max * (gamma * I)
    r_C_inci = dC_incidt - T_max * (sigma * E)

    loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2 + r_C_inci**2)

    
    if causal:
        past_errors = loss.detach()
        #compute number of points
        n_points = past_errors.shape[0]
        cumulative_past_errors = torch.zeros(n_points)
        cumulative_past_errors[1:] = torch.cumsum(past_errors[:-1], dim =0)

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
def ic_loss(model, ICs, T_max, params):
    sigma_array, gamma_array = params
    S0, E0, I0, R0, C_inci0 = ICs
    
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    out = forward_with_constraints(model, t0)
    
    # Value loss - keep gradients!
    pred = out[0, : 5]  # S, E, I, R, C_inci at t =0
    target = torch.tensor([S0, E0, I0, R0, C_inci0], dtype=torch.float32)
    v_loss =  ((pred - target)**2).mean()
   
    # # Derivative loss
    # ones = torch.ones(1)
    # dSdt = grad.grad(out[:,0], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    # dEdt = grad.grad(out[:,1], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    # dIdt = grad.grad(out[:,2], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    # dRdt = grad.grad(out[:,3], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    # dC_incidt = grad.grad(out[:,4], t0, grad_outputs=ones, create_graph=True)[0].squeeze()
    
    # #get sigma and gamma at t=0
    # sigma = torch.tensor(sigma_array[0], dtype=torch.float32)
    # gamma = torch.tensor(gamma_array[0], dtype=torch.float32)
    # Trans = out[0,5] * S0 * I0 # beta * S * I at t=0
   
    # r_S = dSdt +  T_max * Trans
    # r_E = dEdt -  T_max * (Trans - sigma * E0)
    # r_I = dIdt - T_max * (sigma * E0 - gamma * I0)
    # r_R = dRdt - T_max * (gamma * I0)
    # r_C_inci = dC_incidt - T_max * (sigma * E0)
   
    # d_loss = (r_S**2 + r_E**2 + r_I**2 + r_R**2 + r_C_inci**2).mean()
   
    return v_loss

#%%
# data loss
def data_loss(model, t_np, I_obs):
   
    indices = np.arange(0, len(t_np), 7) # every 7th point to reduce computational load, can adjust as needed
    
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
def train_model(epochs=40000, test_days=65, causal = False, epsilon = 5.0, save = False, model_name = 'pinn_model'):
    # general parameters used in the model
    N = 10000
    ICs = [(N-2)/N, 0/N, 0.0, 2/N, 0.0] # no covid cases at the start, seed with 1 or 2 or 10
    t_np_full, t_full, total_days, cases = time_to_train()
    Ir_obs = cases / N # convert cases to a proportion
    
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
      w_ic = min(500, 100 + 0.02 * ep)
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

        # Data loss - Only on training points
        l_data = data_loss(model, t_train_np, Ir_train)
       
       #ODE loss
        if causal:
            l_ode = ode_loss(model, t_train_tensor, params, T_max, epsilon= epsilon, causal= True)
        else:
            l_ode = ode_loss(model, t_train_tensor, params, T_max, epsilon = epsilon, causal = False)

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
        torch.save(model.state_dict(), f"{model_name}.pth") 
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

#%%
def R_eff(out, params):
    sigma, gamma = params
    S    = out[:, 0]   # proportion, no need to multiply by N
    beta = out[:, 5]
    return (beta * S) / gamma   # dimensionless

#Cori et al. (2013) framework used in EpiEstim, 
# with a discretized Gamma serial interval distribution, 
# and a Bayesian posterior for R_t given Poisson incidence data. 
# This is a more traditional epidemiological estimate of R_t that we can compare to the PINN-derived R_t.
def estimate_R_t(cases, mean_si=5.2, sd_si=1.5, window=7, a0=1, b0=2):
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
#comparison of the PINN-derived R_t and the EpiEstim Bayesian R_t estimates,
#  with variant periods shaded, to see how the transmission dynamics evolved over time 
# and how they correspond to the different variant periods in Canada.
def plot_R_t_comparison(out, cases, params, variants, data_start_date):
    cases_np = np.array(cases)
    days = np.arange(len(cases_np))
    
    Rt_pinn = R_eff(out, params)
    Rt_mean, Rt_lower, Rt_upper = estimate_R_t(cases_np)

    # using the same date_to_day function to convert variant period dates to day indices for plotting
    def date_to_day(dt):
        return (dt - data_start_date).days

    plt.figure(figsize=(14, 7))
    variant_colors = ['purple', 'orange', 'green', 'brown']
    plt.plot(days, Rt_pinn, color='orange', linewidth=2, label='R_t (PINN)')
    plt.plot(days, Rt_mean, color='green', linewidth=2, label='R_t (Bayesian)')
    plt.fill_between(days, Rt_lower, Rt_upper, color='grey', alpha=0.3, label='95% CI (Bayesian)')
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='R_t = 1')
    plt.xlabel('Days from data start')
    plt.ylabel('R_t')
    plt.title('Effective Reproduction Number with Variant Periods')
    plt.ylim(0, 5)
    plt.grid(True)
    # Shade variant periods
    for (variant, (start, end)), col in zip(variants.items(), variant_colors):
        s_day = date_to_day(start)
        e_day = date_to_day(end)
        plt.axvspan(s_day, e_day, alpha=0.15, color=col, label=variant)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("R_t_comparison.png")
    plt.close()

#%%
#plot model comparison (simple PINN vs CEIC PINN)
def model_comparison(model1, model2, t):
    t_np_full, t_full, total_days, cases = time_to_train()
    days = np.arange(total_days)
    N = 10000
    sigma_array, gamma_array = get_parama(total_days)

    with torch.no_grad():
        out1 = forward_with_constraints(model1, t).numpy()
        out2 = forward_with_constraints(model2, t).numpy()
    #daily cases and cumulative cases comparison
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(t_np_full, out1[:, 4] * N, label = 'simple PINN')
    plt.plot(t_np_full, out2[:, 4] * N, label = 'CEIC PINN')
    plt.plot(t_np_full, np.cumsum(cases), label = 'observed')
    plt.title('Cumulative Cases Comparison')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Cases')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_np_full[1:], np.diff(out1[:, 4] * N), label = 'simple PINN')
    plt.plot(t_np_full[1:], np.diff(out2[:, 4] * N), label = 'CEIC PINN')
    plt.plot(t_np_full[1:], cases[1:], label = 'observed')
    plt.title('Daily Cases Comparison')
    plt.xlabel('Days')
    plt.ylabel('Daily Cases')
    plt.legend()
    plt.grid(True)
    plt.savefig("model_comparison_simple_CEIC.png")
    plt.close()

    #beta and R_t comparison
    R_t_simple = R_eff(out1, (sigma_array, gamma_array))
    R_t_ceic = R_eff(out2, (sigma_array, gamma_array))
    Rt_mean, Rt_lower, Rt_upper = estimate_R_t(cases, mean_si=5.2, sd_si=1.5, window=5, a0=1.0, b0=5.0)
    

    plt.figure(figsize=(14, 8))
    plt.plot(t_np_full, out1[:, 5], label = 'simple PINN')
    plt.plot(t_np_full, out2[:, 5], label = "CEIC PINN")
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
    pred_test1 = out1[total_days - 65:, 4] * N
    pred_test2 = out2[total_days - 65:, 4] * N
    obs_test = np.cumsum(cases)[total_days - 65:total_days]
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
    N= 10000
    t_np_full, t_full, total_days, cases = time_to_train()
    params = get_parama(total_days)
    num_runs = 1
    #store errors for each run
    simple_error = []
    ceic_error = []
    for run in range(num_runs):
        print("RUN:", run +1)
        set_seed(run)
        print("Training simple PINN__")
        # model_simple, history_s, train_days = train_model(epochs = 50000, test_days= 65, causal= False,
        #                                                 epsilon= 3, save = True, model_name= f"simple_pinn_run_{run+1}")
        print("Training CEIC PINN__")
        # model_ceic, history_c, train_days = train_model(epochs= 50000, test_days= 65, causal = True,
        #                                                epsilon= 3, save = True, model_name= f'ceic_pinn_run_{run+1}')

        #load model
        model_simple = load_model(f"simple_pinn_run_{run+1}.pth")
        model_ceic = load_model(f"ceic_pinn_run_{run+1}.pth")
        with torch.no_grad():
            out_simple = forward_with_constraints(model= model_simple, t = t_full).numpy()
            out_ceic = forward_with_constraints(model= model_ceic, t = t_full).numpy()

            #predictions
            pred_simple_C_inci = out_simple[:, 4] * N
            pred_ceic_C_inci = out_ceic[:, 4] * N
            observed_C_inci = np.cumsum(cases)

            #calculate error on the test set
            test_pred_simple_C_inci = pred_simple_C_inci[-65:]
            test_pred_CEIC_C_inci = pred_ceic_C_inci[-65:]
            test_observed_C_inci = observed_C_inci[-65:]

            mse_simple = np.mean((test_pred_simple_C_inci - test_observed_C_inci) ** 2)
            mse_ceic = np.mean((test_pred_CEIC_C_inci - test_observed_C_inci) ** 2)

            simple_error.append(mse_simple)
            ceic_error.append(mse_ceic)

            #summary
            print(f"RUN {run +1} : Simple PINN MSE: {mse_simple: .3f}, CEIC PINN MSE: {mse_ceic: .3f}")

            #simple_pinn statistics
            simple_mean_error = np.mean(simple_error)
            simple_std_error = np.std(simple_error)
            simple_max_error = np.max(simple_error)
            simple_min_error = np.min(simple_error)
            simple_CI_error = (simple_mean_error - 1.96 * simple_std_error / np.sqrt(num_runs), 
                                simple_mean_error + 1.96 * simple_std_error / np.sqrt(num_runs))
            print(f"Simple PINN MSE - Mean: {simple_mean_error:.3f}, Std: {simple_std_error:.3f}, Max: {simple_max_error:.3f}, Min: {simple_min_error:.3f}, 95% CI: ({simple_CI_error[0]:.3f}, {simple_CI_error[1]:.3f})")

            #CEIC_pinn statistics
            ceic_mean_error = np.mean(ceic_error)
            ceic_std_error = np.std(ceic_error)
            ceic_max_error = np.max(ceic_error)
            ceic_min_error = np.min(ceic_error)
            ceic_CI_error = (ceic_mean_error - 1.96 * ceic_std_error / np.sqrt(num_runs), 
                                ceic_mean_error + 1.96 * ceic_std_error / np.sqrt(num_runs))
            print(f"CEIC PINN MSE - Mean: {ceic_mean_error:.3f}, Std: {ceic_std_error:.3f}, Max: {ceic_max_error:.3f}, Min: {ceic_min_error:.3f}, 95% CI: ({ceic_CI_error[0]:.3f}, {ceic_CI_error[1]:.3f})")
            
            #paired t_test to compare the two models across runs
            t_stat, p_value = ttest_rel(simple_error, ceic_error)
            print(f"Paired t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

            #boxplot comparison
            plt.figure(figsize=(8, 6))
            plt.boxplot([simple_error, ceic_error], tick_labels = ['Simple PINN', 'CEIC PINN'])
            plt.ylabel('MSE')
            plt.title('Comparison of Simple PINN and CEIC PINN')
            plt.grid(True)
            plt.savefig("model_comparison_boxplot.png")
            plt.close()

            #95% confidence interval bar_chart
            #yerr in bar chart means the error bars, which represent the variability of the data, in this case, the confidence interval for the mean MSE of each model. The error bars will extend from the mean value to the upper and lower bounds of the confidence interval, visually showing the range within which we can be 95% confident that the true mean MSE lies for each model.
            plt.figure(figsize=(8, 6))
            plt.bar(['Simple PINN', 'CEIC PINN'], [simple_mean_error, ceic_mean_error], yerr=[simple_CI_error, ceic_CI_error])
            plt.ylabel('Mean MSE')
            plt.title('95% Confidence Interval of MSE')
            plt.grid(True)
            plt.savefig("model_comparison_barplot.png")
            plt.close()

            #plot_R_t_comparison(out_ceic, cases, params, CANADA_VARIANTS, data_start_date = datetime.datetime(2020, 1, 23))
            #plot_all(model_ceic, train_days, t_full)
            model_comparison(model_simple, model_ceic, t_full)

            #"seasonal": seasonal_beta(beta0=0.3, A=0.2, T=180, phase=0),

            
            
        

            #plt.plot(  out_simple[:, 5], label = 'simple_pinn' )
            plt.plot(  out_ceic[:, 5], label = 'ceic_pinn')
            plt.plot( data, label = 'seasonal_beta')
            plt.legend()
            plt.savefig("beta_comp.png")
            plt.close()

            data = get_canada_data()
            plt.plot(data, label= 'observed')
            plt.savefig('observed.png')
            plt.close()


    

