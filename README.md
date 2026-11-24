# CEIC_python_code
#structuaral_recons_backward_work.py
From data_gen.py we generate synthetic data by using SEIuIrRS model by using the ODEs. Then we use the sigma*E = Z(t) as our observed data, and we use a simple feedforward NN to estimate the Z_nn(t) curve. 

Then by using the Z_nn(t) we reconstruct the other compartments. 
 ```
 Z(t) = sigma * E(t) => E(t) = Z(t) / sigma
 I(t) = by solving dI/dt = Z(t) - gamaa * I(t) ---integration
 R(t) = by solving dR/dt = gamma * I(t) - omega * R(t) --- integration
 S(t) = N - E(t) - I(t) - R(t)
 ```

Then for using PINNs we generate five separate networks for S_nn, E_nn, I_nn, R_nn, and Beta_nn, but we train them at the same time by using ODE_loss, data_loss, ic_loss, and struct_loss.

# Requirements
Python 3.13.12
Name: pandas
Version: 3.0.1
Name: numpy
Version: 2.4.3
Name: matplotlib
Version: 3.10.8
Name: torch
Version: 2.8.0

# How to run the code 
Clone the repository-
```bash
git clone https://github.com/nav0111/CEIC_python_code
cd CEIC_python_code
```
Make sure ```data_gen.py``` and ```structuaral_recons_backward_work.py``` are in the same directory.

# Run
```bash
python structuaral_recons_backward_work.py
```

output : Results are saved in `structual_reconstruction_backward_results/`





