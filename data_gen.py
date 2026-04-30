# Generate synthetic data for testing and development purposes 
# Task is to write the relevant functions that generate incidence curves
# from SEIR model and then add noise to it

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

#%%
#generate incidence data from cumulative cases
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rng = np.random.default_rng(seed=627)

def seir_model(t, y, beta_func, sigma, gamma, omega = 0.0):
    S, E, I, R, C = y
    N = S + E + I + R
    beta = beta_func(t)

    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    dC_dt = sigma * E  # Incidence is the rate of new infections
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
    S, E, I, R, Cu_inci = sol.y

    #convert cumulative incidence to daily incidence, prepend the initial cumulative incidence to the beginning of the array to maintain the same length
    inci = np.maximum(np.diff(Cu_inci, prepend=CuIn0), 0.0)
    return S, E, I, R, Cu_inci, inci

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

#function for incidence curve
def get_data(t_arr, beta_func, params, ICs):
    sigma, gamma, omega = params
    S0, E0, I0, R0, CuIn0 = ICs
    inci_t = []
    S, E, I, R, Cu_inci, inci= gen_inci(beta_func, sigma, gamma, omega, S0, E0, I0, R0, CuIn0, t_arr)
    return inci


#%%
if __name__ == "__main__":
    N = 10000
    S0 = N - 2
    E0 = 0
    I0 = 2
    R0 = 0
    CuIn0 = 2
    sigma, gamma, omega = 1/5.2, 1/10, 1/90 #1/180  # incubation period of 5.2 days and infectious period of 10 days
    t = np.arange(366)
    
    #beta functions parameters
    scenarios = { 
        "constant": constant_beta(0.3),
        "sigmoid": sigmoid_beta(beta_low=0.1, beta_high=0.4, k=0.1, t0=180),
        "seasonal": seasonal_beta(beta0=0.3, A=0.2, T=180, phase=0),
        "piecewise": piecewise_beta([0.25, 0.15, 0.4], [120, 240]),
        "sawtooth": sawtooth_beta(beta_min=0.08, beta_max=0.45, t0=0, T=50)
    }    

    beta_curves = {}
    rt_curves = {}
    incidence_curves = {}    
    incidence_data = {}   
    for scenario, beta_func in scenarios.items():
        # save beta curves for plotting later
        beta_curves[scenario] = [beta_func(tval) for tval in t]

        
        #.items() is a dictioanry that loops through both the keys and values of the dictionary, so in this case, scenario will be the name of the beta function with parameters and beta_func will be the corresponding beta function that we defined earlier.
        S, E, I, R, Cu_inci, inci = gen_inci(beta_func, sigma, gamma, omega, S0, E0, I0, R0, CuIn0, t)
        incidence_curves[scenario] = inci

        # add Rt curves for plotting later
        rt_curves[scenario] = calculate_rt(t, beta_func, S, gamma, N)

        #get icidence data
        #incidence_data[scenario] = get_data(t, beta_func)

        #adding noise to the incidence data
        #poisson noise, if inci = [10,20,30], then noisy_inci_poisson= maybe [9, 18, 28] or [11, 22, 32], etc. 
        noisy_inci_poisson = rng.poisson(inci)
        incidence_curves[scenario + "_poisson"] = noisy_inci_poisson
        #negative binomial noise with overdispersion parameter phi = 5
        #negative_binomial(n, p, size=None), n = number of successes, p = probability of success, size = output shape.
        phi_values = [3, 4, 5, 20]  # different values of phi to try
        for phi in phi_values:
            # numpy NB: n=φ (dispersion), p=φ/(φ+μ) gives mean μ, var μ + μ²/φ
            noisy_inci_nb = rng.negative_binomial(n=phi, p=phi/(phi + inci))
            incidence_curves[scenario + f"_nb_phi_{phi}"] = noisy_inci_nb
    
    
    # plotting beta curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(t, beta_curves["constant"])
    axes[0, 1].plot(t, beta_curves["sigmoid"])
    axes[0, 2].plot(t, beta_curves["seasonal"])
    axes[1, 0].plot(t, beta_curves["piecewise"])
    axes[1, 1].plot(t, beta_curves['sawtooth'])
    plt.savefig("beta_curves.png")
    plt.close()

    #plotting the incidence curves with negative binomial noise 
    plt.figure(figsize=(15, 10))
    #with enumerate gives both index and value
    for i, phi in enumerate(phi_values):
        plt.subplot(2,2,i+1)
        plt.plot(t, incidence_curves["constant_nb_phi_" + str(phi)], label=f"Constant Beta with NB Noise (phi={phi})")
        plt.plot(t, incidence_curves["sigmoid_nb_phi_" + str(phi)], label=f"Sigmoid Beta with NB Noise (phi={phi})")
        plt.plot(t, incidence_curves["seasonal_nb_phi_" + str(phi)], label=f"Seasonal Beta with NB Noise (phi={phi})")
        plt.plot(t, incidence_curves["piecewise_nb_phi_" + str(phi)], label=f"Piecewise Beta with NB Noise (phi={phi})")
        plt.plot(t, incidence_curves["sawtooth_nb_phi_" + str(phi)], label=f"Sawtooth Beta with NB Noise (phi={phi})")
        plt.title(f"Negative Binomial Noise with phi={phi}")
        plt.xlabel("Time (days)")
        plt.ylabel("Daily Incidence")
        plt.legend()
    plt.savefig("incidence_curves_betas_with_nb_noise.png")
    plt.close()

    # #plot with poisson noise
    plt.figure(figsize=(15,10))
    plt.plot(t, incidence_curves['constant'], 'b-' , label= 'constant_beta ')
    plt.plot(t, incidence_curves['sigmoid'], 'g-', label = 'sigmoid_beta')
    plt.plot(t, incidence_curves['seasonal'], 'r-',label = "seasonal_beta")
    plt.plot(t, incidence_curves["piecewise"], 'k-',label = "piecewise_beta")
    plt.plot(t, incidence_curves["sawtooth"], 'o-',label = "sawtooth_beta")
    plt.plot(t, incidence_curves["constant_poisson"], 'b--' , label = "constant_beta with poisson noise")
    plt.plot(t, incidence_curves["sigmoid_poisson"], 'g--',label = "sigmoid_beta with poisson noise")
    plt.plot(t, incidence_curves["seasonal_poisson"], 'r--',label = "seasonal_beta with poisson noise")
    plt.plot(t, incidence_curves["piecewise_poisson"], 'k--',label ="piecewise_beta with poisson noise")
    plt.plot(t, incidence_curves["sawtooth_poisson"], 'o--',label = "sawtooth beta with poisson noise")
    plt.title("incidence curve of diff bets with poisson noise")
    plt.xlabel("Time (days)")
    plt.ylabel("Daily incidence")
    plt.legend()
    plt.savefig("incidence_curves_with_poisson_noise.png")
    plt.close()

    # plot Rt curves
    plt.figure(figsize=(15, 10))
    plt.plot(t, rt_curves["constant"], label="Constant Beta")
    plt.plot(t, rt_curves["sigmoid"], label="Sigmoid Beta")
    plt.plot(t, rt_curves["seasonal"], label="Seasonal Beta")
    plt.plot(t, rt_curves["piecewise"], label="Piecewise Beta")
    plt.plot(t, rt_curves["sawtooth"], label="Sawtooth Beta")
    plt.title("Effective Reproduction Number (R_t) Curves")
    plt.xlabel("Time (days)")
    plt.ylabel("R_t")
    plt.legend()
    plt.savefig("rt_curves.png")
    plt.close()

    #get incidence data for a specific beta
    incidence = get_data(t, sigmoid_beta(beta_low=0.1, beta_high=0.4, k=0.1, t0=180), [sigma, gamma, omega], [S0, E0, I0, R0, CuIn0])
    
    print(incidence)
    print(len(incidence))
