# CEIC_python_code
#structuaral_recons_backward_work.py
From data_gen.py we generate synthetic data by using SEIuIrRS model by using the ODEs. Then we use the sigma*E = Z(t) as our observed data, and we use a simple feedforward NN to estimate the Z_nn(t) curve. 

Then by using the Z_nn(t) we reconstruct the other compartments. 
# Z(t) = sigma * E(t) => E(t) = Z(t) / sigma
# I(t) = by solving dI/dt = Z(t) - gamaa * I(t) ---integration
# R(t) = by solving dR/dt = gamma * I(t) - omega * R(t) --- integration
# S(t) = N - E(t) - I(t) - R(t)

Then for using PINNs we generate five separate networks for S_nn, E_nn, I_nn, R_nn, and Beta_nn, but we train them at the same time by using ODE_loss, data_loss, ic_loss, and struct_loss.

