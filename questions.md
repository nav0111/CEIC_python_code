Questions for Akhi to answer. 
AI is strictly forbidden. 

1. This function creates a NN with one input. 
```
def create_model():
    model = nn.Sequential(
        nn.Linear(1, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 5)
    )
    return model
```
- What is the `type` of the input? 
- How do you create a variable of this type. 
- Write 3 - 4 lines that runs the model with input $t = 2$. 

reference: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

https://www.geeksforgeeks.org/python/tensors-in-pytorch/

1. What does the function `np.full()` do? 

1. Understanding AD on tensors. 
```
x = torch.tensor(3.0, requires_grad=True)
y = x * 2
print(y)
```
Ooutput is `tensor(6., grad_fn=<MulBackward0>)`.
The computation `y = x * 2` is an operation on `y`, so pytorch attaches a gradient function describing how to backprop through multiplication.

Run this code: 
```
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * 2
z = y + 5

print(z)
print(z.grad_fn)
print(z.grad_fn.next_functions)
``` 
Look at the print statements. The `computation graph` is as follows: 
```
x --(MulBackward0)--> y --(AddBackward0)--> z
```
We can take derivative! Try `z.backward()` and print out `print(x.grad)`. What is the result?

Since $z = (x \cdot 2) + 5$ which implies $dz = 2$. 

Try the computation graph for your entire network: 
```
import torch
import torch.nn as nn

def create_model():
    return nn.Sequential(
        nn.Linear(1, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 5)
    )

model = create_model()
x = torch.tensor([[1.0]], requires_grad=True)
y = model(x)

print(y)
print(y.grad_fn)
print(y.grad_fn.next_functions)
```
output: 
```
tensor([...], grad_fn=<ViewBackward0>)
((<AddmmBackward0 object at ...>, 0),)
```

you can print the entire computational graph: 
```
def print_graph(fn, indent=0):
    print(" " * indent, fn)
    if hasattr(fn, "next_functions"):
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                print_graph(next_fn, indent + 4)

print_graph(y.grad_fn)
```

So if we want to compute $\frac{dy}{dx}$ (and since $y$ is a 5-dimensional vector), we get a gradient vector (or really a $5 times 1$ Jacobian matrix). 

Try this to backprop one output at a time
```
for i in range(5):
    model.zero_grad()
    y[i].backward(retain_graph=True)
    print(f"dy[{i}]/dx =", x.grad.item())
```
or
```
from torch.autograd.functional import jacobian

J = jacobian(lambda inp: model(inp), x)
print(J)
```

Why do you need the `retain_graph` there? 

PyTorch frees the computation graph immediately after backprop.
```
backward() → compute gradients → delete graph
```
We are calling `backward()` five times (manually) so we need to maintain the computation graph to be able to calculate the derivatives. Try the counter example: 

```
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * 2

y.backward()
y.backward()
```

1. Can you explain why `tanh` is used? I undersstand it's an infinitely differentiable function, and that RelU has a kink in it (plus zero second derivative). Is it because in the PINN we are taking the derivative of the NN twice with respect to the weights/biases and also time? Can you find sources/papers which show why RelU fails in PINNs. 

For PINN the requirement is smooth (C^∞ ideally), non-saturating in the active region, derivatives don't vanish under composition. Tanh hits this. There are other candidates that hit it differently.

The initialization of the network really matters for the activation function also. 

Main point: PINNs for inverse problems have many degenerate local minima, and getting out of them requires either good initialization or curriculum learning.  The Wang causality paper sidesteps this issue because they're doing forward PDE problems where the IC is known and the trick of "learn early times first" automatically gives the network something non-trivial to fit.

The PINN being stuck in the degenerate basin has to do with gradient pathways (the softplus problem), initialization basins (β starting at 0.69), and loss balance (data needs to dominate during warmup).


== Pathologies in inverse problems with SEIR. 
If we fit to $I(t)$ from the SEIRS model, then $\beta(t)$ is fully identifiable. How? Given $I(t)$, we know $dI/dt$. From there we can get $E$ since $\sigma E = dI/dt + \gamma I$ so $E$ is determined. From the last equation we have $dR/dt + \omega R = \gamma I$ which is a linear ODE solvable, so $R(t)$ is known as well. From the conservation constraint, S is fully determined. Then finally from $\beta S I = dE/dt + \sigma E$, we can fully determine $\beta(t)$. 

So the in verse problem is well posed. This is structurally identifiable.

Failure mode 1: the network doesn't fit $I(t)$. 
Failure mode 2: derivatives are noisy when computed from a network. Even if the network fits I(t) well in value-space, the chain I → dI/dt → E → dE/dt → β is unstable. Each differentiation amplifies small errors. The PINN's I matches I_true well in value but its dI/dt may differ from the true dI/dt by terms that look small (1e-3) but are large relative to the actual derivative magnitudes. So even if I_network ≈ I_true, you get E_network ≠ E_true downstream, then β_network ≠ β_true. This is a numerical conditioning problem of the inverse derivative chain, not the network architecture.

Failure mode 3: soft ODE constraints leave β with slack. The math above assumes the ODE residual is exactly zero. In a PINN, the residual is minimized as a soft constraint. If w_ode is small relative to w_data, the network can produce an (S, E, I, R, β) tuple where:
- I matches data perfectly 
- the ODE residual is small 
- But β·S·I differs from dE/dt + σ·E by amounts that translate to large β errors. 
The ODE residual loss is mean(r²) and is ≈ 1e-8. In log-space this looks excellent. In actual implications for β, it's enormous slack.

Why CEIC don't help: The reason CEIC didn't help and Wang-causal didn't help: both modify the ODE residual computation. But the ODE residual is the very thing being underweighted. Modifying how it's computed when it's already minor doesn't move the needle. Causal weighting can also actively hurt here — it down-weights residuals at later times if earlier times have any error, which exacerbates the slack at the late-time portion where seasonal β has its second peak.


Reason 1: The data loss is a direct constraint; the β loss is an indirect inference

The data loss is (I_pred - I_obs)² — directly comparing the network's I output to the target. Every gradient step on the data loss updates the parameters that affect I_pred directly. Fast, strong, clean signal.
There is no direct loss on β. The network's β output is only constrained through the ODE residual, specifically the E equation: dE/dt + σE - β·S·I = 0, which rearranges to β = (dE/dt + σE) / (S·I).
So β is only being learned through a chain:

The ODE residual must be small.
For the residual to be small, β must be consistent with (dE/dt + σE) / (S·I).
For this to identify the true β, the network's E, S, I must match the true E, S, I.
For E, S, I to match the truth, the ODE residual must be small everywhere, plus the data must constrain I, plus conservation must hold.

This chain is fragile. Each step depends on the previous, and errors compound. Meanwhile, the data loss is one step: I_pred → loss. No chain, no error compounding.
This asymmetry is structural — it's true of any PINN inverse problem with soft constraints, not just yours.


Reason 2: Many β trajectories satisfy the ODE residual within tolerance
When the ODE residual is small (say r_E² ≈ 1e-7), there isn't a unique β — there's a family of β's compatible with the network's E, S, I trajectories. The optimizer will pick the smoothest, simplest β from this family, because:
- The neural network's output is naturally smooth (tanh + finite weights = smooth function).
- The simplest function the β-head can output is approximately constant.
- Once the network has found any β trajectory in the residual-compatible family, there's no pressure to move within that family.
