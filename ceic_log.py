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
import data_gen

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

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
#Canada Data
# def get_canada_data():
#     df = pd.read_csv('cases_can.csv')  
#     df['date'] = pd.to_datetime(df['date'])
#     inc_cases = df['value_daily'].rolling(window=7).mean().dropna()
#     return inc_cases

#%%
#Variant periods in Canada based on public health data and reports, with start and end dates for each variant's dominance
# CANADA_VARIANTS = {
#     'Alpha':   (datetime.datetime(2020, 12, 27), datetime.datetime(2021, 6, 30)),
#     'Delta':   (datetime.datetime(2021, 7, 1),   datetime.datetime(2021, 12, 14)),
#     'Omicron': (datetime.datetime(2021, 12, 15), datetime.datetime(2022, 5, 31)),
#     'BA.5':    (datetime.datetime(2022, 6, 1),   datetime.datetime(2023, 1, 31)),
# }

# #%%
# #Variant parameters (sigma and gamma) based on literature estimates of incubation and infectious periods for each variant, converted to rates (1/days)
# VARIANT_PARAMS = {
#     'Alpha': (1/5.2, 1/10),  # sigma, gamma
#     'Delta': (1/4.5, 1/8),
#     'Omicron': (1/3.5, 1/6),
#     'BA.5': (1/3.0, 1/5)
# }

# #%%

# def get_parama(total_days):
#     #sigma and gamma parameters for the variants in covid
#     start_date = datetime.datetime(2020, 1, 23)

#     #create empty arrays for sigma and gamma 
#     sigma_t = np.zeros(total_days)
#     gamma_t = np.zeros(total_days)

#     #loop through each day and assign sigma and gamma based on the variant period
#     for i in range(total_days):
#         current_date = start_date + datetime.timedelta(days=i)
#         for variant, (start, end) in CANADA_VARIANTS.items():
#             if start <= current_date <= end:
#                 sigma_t[i], gamma_t[i] = VARIANT_PARAMS[variant]
#                 break
#         else:
#             sigma_t[i], gamma_t[i] = VARIANT_PARAMS['Alpha'] # default to Alpha parameters if no variant matches

#     params = sigma_t, gamma_t
#     return params

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
#PINN model
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 5)
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
    beta = F.softplus(raw[:, 4:5])
    return torch.cat([S, E, I, R, beta], dim=1)

# #define daily incidence
# def daily_incidence(model, t):
#     out = forward_with_constraints(model, t)
#     E = out[:, 1]
#     sigma_array, gamma_array = get_parama(len(t))
#     sigma = torch.tensor(sigma_array, dtype = torch.float32)
#     daily_inci = sigma * E
#     return daily_inci


#%%
#ODE loss
def ode_loss(model, t, params, epsilon, causal = False):
    sigma_array, gamma_array = params
    out = forward_with_constraints(model, t)
   
    # the output of the neural network at time t
    S = out[:, 0]
    E = out[:, 1]
    I = out[:, 2]
    R = out[:, 3]
    beta = out[:, 4]

   #creates a tensor of all 1's same shape as S, need this to combine gradients from multiple outputs
    ones = torch.ones_like(S)
    #create_graph= True, because we need derivative of a derivative otherwise the term would consider as a constant
   
    dSdt = grad.grad(S, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dEdt = grad.grad(E, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dIdt = grad.grad(I, t, grad_outputs=ones, create_graph=True)[0].squeeze()
    dRdt = grad.grad(R, t, grad_outputs=ones, create_graph=True)[0].squeeze()
   
    #convert sigma and gamma arrays to tensors and get the values at the corresponding time points
    sigma = torch.tensor(sigma_array, dtype=torch.float32)
    gamma = torch.tensor(gamma_array, dtype=torch.float32)
    #force of infection term
    Trans = beta * S * I # beta * S * I

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
    pred = out[0, : 4]  # S, E, I, R at t =0
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
    C = sigma * E 
    # dEdT = beta S I - C
    # C = dEdT - beta S I 
    ones = torch.ones_like(E)
    dEdt = grad.grad(E, t_k, grad_outputs=ones, create_graph=True)[0].squeeze()
    C =  beta * S * I - dEdt

    v_loss = torch.mean((C - observed) **2)
    return v_loss, C


#%%
#Time to train the model
def time_to_train():
    cases = get_canada_data()
    total_days = len(cases)
    t_np_full = np.linspace(0, total_days, total_days, dtype=np.float32)
    t_full = torch.tensor(t_np_full.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    return t_np_full, t_full, total_days
#%%
#Train model
def train_model(epochs=40000, test_days=65, causal = False, epsilon = 3, save = False, model_name = 'pinn_model'):
    # general parameters used in the model
    N = 100000
    ICs = [(N-2)/N, 0/N, 2/N, 0/N] # no covid cases at the start, seed with 1 or 2 or 10
    t_np_full, t_full, total_days = time_to_train()
    cases = get_canada_data()
    Ir_obs = cases / N # convert cases to a proportion
    
    # Split into train and test
    train_days = total_days - test_days
    train_indices = np.arange(0, train_days)
    
    # Training time tensor (only training points)
    t_train_np = t_np_full[train_indices] # training time
    t_train_tensor = torch.tensor(t_train_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True) 
    Ir_train = Ir_obs[train_indices]      # and cases at those time points

    print(f"t_train_np: {t_train_np[:20].flatten()}")
    print(f"t_train_tensor: {t_train_tensor[:20].flatten()}")
    
    #time_varying parameters for the variants
    params_full = get_parama(total_days)
    sigma_array, gamma_array = params_full
    params = sigma_array[train_indices], gamma_array[train_indices] # only training period parameters

    # Create model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    #weights for different loss compartments
    def get_weights(ep):
        w_ic = 100  
        w_ode = 1.0  # Keep ODE weight constant
        w_data = 1000   #
        return w_ic, w_ode, w_data

    print(f"{'Ep':>6} {'IC':>12} {'ODE':>12} {'data':>12} {'Total':>12}  S_range       beta_range")
   
    history = []
    for ep in range(epochs):
        w_ic , w_ode, w_data = get_weights(ep)
        optimizer.zero_grad() #reset any previous gradients
        
        l_ic = ic_loss(model, ICs)
        l_data, C = data_loss(model, t_train_np, Ir_train, sigma_array[train_indices])

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
    return model, history, train_days, C

#%%
#Plot
def plot_all(model, train_days,t_full):
    N = 100000
    t_np_full, t_full, total_days = time_to_train()
    cases = get_canada_data()
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
def R_eff(out, params):
    sigma, gamma = params
    S    = out[:, 0]   # proportion, no need to multiply by N
    beta = out[:, 4]
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

    #beta and R_t comparison
    R_t_simple = R_eff(out1, (sigma_array, gamma_array))
    R_t_ceic = R_eff(out2, (sigma_array, gamma_array))
    Rt_mean, Rt_lower, Rt_upper = estimate_R_t(cases, mean_si=5.2, sd_si=1.5, window=5, a0=1.0, b0=5.0)
    

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
    N= 100000
    num_runs = 1
    #store errors for each run
    simple_error = []
    ceic_error = []

    #store beta for each run to see the variability across runs
    all_beta_simple = []
    all_beta_ceic = []
    
    for run in range(num_runs):
        t_np_full, t_full, total_days = time_to_train()
        cases = get_canada_data()
        params = get_parama(total_days)
        sigma_array, gamma_array = params
        print("RUN:", run +1)
        #set_seed(run)
        
        print("Training simple PINN__")
        model_simple, history_s, train_days, C_simple = train_model(epochs = 20000, test_days= 65, causal= False,
                                                         epsilon = 3, save = True, model_name= f"simple_pinn_run_{run+1}")
        print("Training CEIC PINN__")
        model_ceic, history_c, train_days, C_ceic = train_model(epochs= 20000, test_days= 65, causal = True,
                                                       epsilon = 3, save = True, model_name= f'ceic_pinn_run_{run+1}')

        #load model
        # model_simple = load_model(f"simple_pinn_run_{run+1}.pth")
        # model_ceic = load_model(f"ceic_pinn_run_{run+1}.pth")
        with torch.no_grad():
            out_simple = forward_with_constraints(model= model_simple, t = t_full).numpy()
            out_ceic = forward_with_constraints(model= model_ceic, t = t_full).numpy()

            beta_simple = out_simple[:, 4]  # beta is column 4
            beta_ceic   = out_ceic[:, 4]

            all_beta_simple.append(beta_simple)
            all_beta_ceic.append(beta_ceic)

        #model_comparison(model_simple, model_ceic, t_full)

            #calling observed beta
        true_beta = data_gen.seasonal_beta(beta0=0.3, A=0.2, T=180, phase=0)(t_np_full)
        #true_beta = data_gen.piecewise_beta([0.1, 0.15, 0.25, 0.3, 0.4], [60,120, 240, 300])(t_np_full)
        #plot beta variability across runs for CEIC and true beta and simple PINN
        plt.figure(figsize=(14, 8))
        for i, beta in enumerate(all_beta_ceic):
            plt.plot(t_np_full, beta, color='blue', label='CEIC PINN' if i == 0 else "")
        for i, beta in enumerate(all_beta_simple):
            plt.plot(t_np_full, beta, color='green', label='Simple PINN' if i == 0 else "")
        plt.plot(t_np_full, true_beta, label = 'True Beta', color = 'red')
        plt.title('Beta Variability Across Runs (CEIC PINN)')
        plt.xlabel('Days')
        plt.ylabel('Beta')
        plt.legend()
        plt.grid(True)
        plt.savefig("beta_variability_ceic_and_simple.png")
        plt.close()


            

        



    

