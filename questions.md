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

2. What does the function `np.full()` do? 

3. Understanding AD on tensors. 
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

