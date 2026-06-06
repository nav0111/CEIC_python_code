# Generate synthetic data for testing and development purposes 
# Task is to write the relevant functions that generate incidence curves
# from SEIR model and then add noise to it
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Your SEIR model should include a time-dependent beta function
# Let's do 5 different scenarios for beta(t):
# for each of the following, pick reasonable parameters for the functions 
# and generate the corresponding incidence curves:
#  - Constant beta: β(t) = β0
#  - Sigmoid beta function: β(t) = β_low + (β_high - β_low) / (1 + exp(-k * (t - t0)))
#  - Seasonal β(t) = β0 * (1 + A * sin(2 * π * t / T  + phase)) 
#    T is the period, so I think we can pick T = 365 here. 
#  - Piece-wise constant: β(t) = ∑ᵢ₌₀ⁿ⁻¹ βᵢ  𝟏₍ₜᵢ,  ₜᵢ₊₁₎(t)
#  - Saw tooth: β(t) = βₘᵢₙ + (βₘₐₓ - βₘᵢₙ) ⋅ ((t - t₀)mod T) / T
#    this is a  linear rise then sudden drop. 
#    you can also turn this into a smoother function by I guess taking the Fourier series

# The SEIR model should also include ρ(t) term that represents the reporting rate of cases
# if ρ(t) = 1, perfect reporting, if ρ(t) < 1, underreporting of cases
# For now, let's just assume ρ(t) is constant, 
# but we can also make it time-dependent in the future.

# Adding noise to the data: 
# When you generate the incidence curves, add noise to simulate real-world data.
# Noise levels 
# - Poisson: Yₜ ∣ μₜ ~ Poisson(μₜ), where you can have μₜ as the incidence data at time t
# - Negative Binomial: Yₜ ∣ μₜ ~ NegBin(μₜ, φ),   Var(Yₜ) = μₜ + μₜ²/φ
#  try with different values of φ (say 3 - 5) or φ = 20 for overdispersion

# run this in a cell for interactivity. 
# %reload_ext autoreload
# %autoreload 2

#%%
#generate incidence data from cumulative cases
def seir_model(t, y, beta_func, sigma, gamma, omega = 0.0):
    S, E, I, R, C = y
    N = S + E + I + R
    beta = beta_func(t)

    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    dC_dt = sigma * E  # incidence per unit time is the rate of new infections
    # solving SEIR means integration dC/dt which gives cumulative cases.
    return [dSdt, dEdt, dIdt, dRdt, dC_dt]
#%%
# beta functions
#constant beta
#because the ODE solver solve_ivp function expects a function for beta(t), we need to define a function that returns a constant value for beta at any time t. 
def constant_beta(beta0):
    # this is called a closure
    def beta_func(t):
        return beta0
    return beta_func

#sigmoid beta function: beta(t) = β_low + (β_high - β_low) / (1 + exp(-k * (t - t0)))
#beta(t) is the transmission rate at time t, β_low is initial transmission rate, 
# β_high is the final transmission rate, k is the growth rate of the sigmoid, 
# and t0 is the time at which the transmission rate is halfway between β_low and β_high.
def sigmoid_beta(beta_low, beta_high, k, t0):
    def beta_func(t):
        beta = beta_low + (beta_high - beta_low) / (1 + np.exp(-k * (t - t0)))
        return beta
    return beta_func

#seasonal beta function: β(t) = β0 * (1 + A * sin(2 * π * t / T  + phase))
#β0 is the baseline transmission rate, A is the amplitude of seasonal variation,
#T is the period of the seasonal variation (e.g., 365 days for yearly seasonality),
#and phase is the phase shift of the seasonal variation.

def seasonal_beta(beta0, A, T, phase):
    def beta_func(t):
        beta = beta0 * (1 + A * np.sin(2 * np.pi * t / T + phase))
        return beta
    return beta_func

#piece-wise constant beta function: β(t) = β0 if t < t1
#                                    β(t) = β1 if t1 <= t < t2
 #                                   β(t) = β2 if t2 <= t < t3, etc.

def piecewise_beta(beta_values, change_times):
    beta_values = np.asarray(beta_values, dtype=float)
    change_times = np.asarray(change_times, dtype=float)
    assert len(beta_values) == len(change_times) + 1
    def beta_func(t):
        idx = np.searchsorted(change_times, t, side="right")
        return beta_values[idx]
    return beta_func

#saw tooth beta function: β(t) = beta_min + (beta_max - beta_min) * ((t - t0) % T) / T
#beta_min is the minimum transmission rate, beta_max is the maximum transmission rate,
#t0 is the time at which the sawtooth function starts, and T is the period of the sawtooth function.
#sawtooth pattern is the transmission rate that increases linearly from beta_min to 
# beta_max over a period of T, then drops back to beta_min and repeats the cycle.

def sawtooth_beta(beta_min, beta_max, t0, T):
    def beta_func(t):
        beta = beta_min + (beta_max - beta_min) * ((t - t0) % T) / T
        return beta
    return beta_func

#%%
#generate incidence data from cumulative cases
def gen_inci(beta_func, sigma, gamma, omega, S0, E0, I0, R0, CuIn0, t):
    y0 = [S0, E0, I0, R0, CuIn0] 
    #solve the SEIR model using solve_ivp, which is a numerical solver for ODE.
    #solve_ivp(fun, t_span, y0, args=(), t_eval=None, method='RK45', vectorized=False)),fun is the function that defines the system of ODEs, t_span is the time span for integration,y0 is the initial conditions, args are additional arguments to pass to the function like beta_func,sigma, gamma, t_eval is the time points at which to store the computed solution, method is the integration method to use (default is 'RK45'), and vectorized indicates whether fun is implemented in a vectorized fashion.
    sol = solve_ivp(fun=seir_model, t_span=[t[0], t[-1]], y0=y0, args=(beta_func, sigma, gamma, omega), 
                    t_eval=t, max_step= 0.1) # max_step is set to 0.1 to ensure we capture the dynamics accurately, especially if beta changes rapidly
    S, E, I, R, C = sol.y
    #convert cumulative incidence to daily incidence, prepend the initial cumulative incidence to the beginning of the array to maintain the same length
    Z = np.maximum(np.diff(C, prepend=CuIn0), 0.0)
    return S, E, I, R, C, Z

#%% 
# calculate Rt, # t is time array
def calculate_rt(t_arr, beta_func, susc_arr, N, gamma):
    # Instantaneous effective reproduction number from Fraser 2007
    # R(t) = β(t) / γ · S(t) / N 
    R_t = [] # create empty list to store the R_t values
    # loop through each time value 
    for t in t_arr: 
        R_t.append(beta_func(t) / gamma * susc_arr[t] / N) 
    return R_t

#%% 
# generate data with defaults
def generate_data_with_defaults(beta_func): 
    # function creates a panel of all the scenarios 
    t = np.arange(730) 
    N = 100000
    S0 = N - 2
    E0 = 0
    I0 = 2
    R0 = 0
    C0 = 0 ### ? why I guess, initial infections. Let's start with 2 exposed. 
    sigma, gamma, omega = 1/5.2, 1/10, 1/60 # no waning immunity for now, so omega = 0.0
    _, _, _, _, _, Z = gen_inci(beta_func, sigma, gamma, omega, S0, E0, I0, R0, C0, t)
    return Z

if __name__ == "__main__":
    # code that will run when we execute this file directly, 
    # but not when we import it as a module in another file.

    # TO DO : find the reproduction number for each of these, but remember 
    # beta(t) is temporal, so need to find a way to do this -- research this. 
    # need this for your thesis/paper: report beta(t), shape/formula, the incidence curve, R_effective
    beta_fn1 = constant_beta(0.11)
    beta_fn2 = sigmoid_beta(beta_low=0.10, beta_high=0.20, k=0.08, t0=400)
    beta_fn3 = seasonal_beta(beta0=0.14, A=0.3, T=365, phase=0)
    beta_fn4 = piecewise_beta(beta_values=[0.22, 0.10, 0.16], change_times=[120, 360])
    beta_fn5 = sawtooth_beta(beta_min=0.10, beta_max=0.20, t0=0, T=180)
    data1 = generate_data_with_defaults(beta_fn1) 
    data2 = generate_data_with_defaults(beta_fn2) 
    data3 = generate_data_with_defaults(beta_fn3) 
    data4 = generate_data_with_defaults(beta_fn4) 
    data5 = generate_data_with_defaults(beta_fn5) 
    print(f"final size: {data1.sum()}")
    print(f"final size: {data2.sum()}")
    print(f"final size: {data3.sum()}")
    print(f"final size: {data4.sum()}")
    print(f"final size: {data5.sum()}")
    
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 5, 1)
    plt.plot(data1, label = 'constant beta')
    plt.legend()

    plt.subplot(2, 5, 2)
    plt.plot(data2, label = 'sigmoid beta')
    plt.legend()

    plt.subplot(2, 5, 3)
    plt.plot(data3, label = 'seasonal beta')
    plt.legend()

    plt.subplot(2, 5, 4)
    plt.plot(data4, label = 'piecewise beta')
    plt.legend()

    plt.subplot(2, 5, 5)
    plt.plot(data5, label = 'sawtooth beta')
    plt.legend()

    plt.subplot(2, 5, 6)
    beta_val = [beta_fn1(x) for x in range(730)]
    plt.plot(beta_val, label = 'constant beta')
    plt.legend()

    plt.subplot(2, 5, 7)
    plt.plot(beta_fn2(np.arange(730)), label = 'sigmoid beta')
    plt.legend()

    plt.subplot(2, 5, 8)
    plt.plot(beta_fn3(np.arange(730)), label = 'seasonal beta')
    plt.legend()

    plt.subplot(2, 5, 9)
    plt.plot(beta_fn4(np.arange(730)), label = 'piecewise beta')
    plt.legend()

    plt.subplot(2, 5, 10)
    plt.plot(beta_fn5(np.arange(730)), label = 'sawtooth beta')
    plt.legend()
    plt.savefig('Five_incidence_curves.png')
    plt.close()

    

    

       
    
    





