# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
# %%
# batch of 8 points
t = torch.rand(8)
X = torch.rand((8, 2))
# weights of a neuron with 2 inputs
w = torch.rand(2)

print(X.shape)
print(X)
print(w.shape)
print(w)

# %%
# neuron state
s0 = X @ w
print(s0.shape)
print(s0)

# %%
# activation functions:
β = -6
# β = -2
# β = -20
σ = lambda s: 1.0 / (1 + torch.exp(β * s))

# %%
y = σ(s0)
print(y.shape)
print(y)

# %%
# augmenting vectors
θ = torch.tensor([0.5])

X_ = torch.cat((torch.ones(8, 1) * -1, X), dim=-1)
print(X_.shape)
print(X_)

w_ = torch.cat((θ, w), dim=-1)
w_.requires_grad = True
print(w_.shape)
print(w_)

# %%
s_ = X_ @ w_
y_ = σ(s_)
print(y_.shape)
print(y_)

# %%
print(w_.grad)

loss = F.mse_loss(y_, t)
loss.backward()

print(w_.grad)

# %%
# t = torch.rand(8)
# X = torch.rand((8, 3))
weights = torch.rand(3, requires_grad=True)
for idx in range(300):
    state = X_ @ weights
    pred = σ(state)
    loss = F.mse_loss(pred, t)
    loss.backward()
    if (idx % 30) == 0:
        # print(f"{idx} W: ", weights)
        print(f"{idx} LOSS: ", loss)
        # print(f"{idx} GRAD: ", weights.grad)

    with torch.no_grad():  # this detaches weights from graph
        gradient = weights.grad
        weights = weights - 0.1 * gradient

    weights.requires_grad_()
    weights.grad = None


# %%
