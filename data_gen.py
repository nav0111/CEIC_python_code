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
