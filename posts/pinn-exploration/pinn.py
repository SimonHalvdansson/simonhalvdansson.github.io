import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ---------------------------
# Define the PINN model
# ---------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=128, num_layers=3, activation=nn.Tanh()):
        """
        A configurable MLP for the Schrödinger equation.
        - input_dim: dimension of input (x,t)
        - output_dim: dimension of output (u,v)
        - hidden_dim: number of neurons per hidden layer
        - num_layers: number of hidden layers
        - activation: activation function (e.g. nn.Tanh() or nn.ReLU())
        """
        super(PINN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Domain settings and hyperparameters
# ---------------------------
# Spatial domain: x in [0,1], time domain: t in [0,1]
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Network hyperparameters (easy to change)
hidden_dim = 128
num_layers = 3
activation = nn.Tanh()  # You can change to nn.ReLU() if desired

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Generate training data
# ---------------------------
# 1. Collocation points for the PDE residual (inside the domain)
N_f = 10000
x_f = torch.rand(N_f, 1, device=device) * (x_max - x_min) + x_min
t_f = torch.rand(N_f, 1, device=device) * (t_max - t_min) + t_min
X_f = torch.cat([x_f, t_f], dim=1)
X_f.requires_grad = True

# 2. Boundary points (x=0 and x=1, with t in [0,1])
N_b = 200
t_b = torch.rand(N_b, 1, device=device) * (t_max - t_min) + t_min
x_b0 = torch.full((N_b, 1), x_min, device=device)
x_b1 = torch.full((N_b, 1), x_max, device=device)
X_b0 = torch.cat([x_b0, t_b], dim=1)
X_b1 = torch.cat([x_b1, t_b], dim=1)
X_b = torch.cat([X_b0, X_b1], dim=0)  # both boundaries

# 3. Initial condition points (t=0, x in [0,1])
N_i = 200
x_i = torch.rand(N_i, 1, device=device) * (x_max - x_min) + x_min
t_i = torch.full((N_i, 1), t_min, device=device)
X_i = torch.cat([x_i, t_i], dim=1)
# Initial condition: psi(x,0)= sin(pi*x) => u(x,0)= sin(pi*x), v(x,0)=0
u_i = torch.sin(np.pi * x_i)
v_i = torch.zeros_like(x_i)

# ---------------------------
# Instantiate the model, optimizer, and loss function
# ---------------------------
model = PINN(input_dim=2, output_dim=2, hidden_dim=hidden_dim,
             num_layers=num_layers, activation=activation).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

# ---------------------------
# Prepare to track losses
# ---------------------------
total_losses = []
pde_losses = []
bc_losses = []
ic_losses = []

# ---------------------------
# Training loop with tqdm live loss in scientific notation
# ---------------------------
num_epochs = 5000
pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    optimizer.zero_grad()
    
    # --- PDE Residual Loss ---
    pred_f = model(X_f)
    u_f = pred_f[:, 0:1]
    v_f = pred_f[:, 1:2]
    
    # Compute gradients with respect to (x, t)
    grad_u = torch.autograd.grad(u_f, X_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    
    grad_v = torch.autograd.grad(v_f, X_f, grad_outputs=torch.ones_like(v_f), create_graph=True)[0]
    v_x = grad_v[:, 0:1]
    v_t = grad_v[:, 1:2]
    
    # Second derivatives with respect to x
    u_xx = torch.autograd.grad(u_x, X_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    v_xx = torch.autograd.grad(v_x, X_f, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    
    # Schrödinger PDE residuals (split into real and imaginary parts):
    # u_t + 0.5*v_xx = 0   and   v_t - 0.5*u_xx = 0
    res_u = u_t + 0.5 * v_xx
    res_v = v_t - 0.5 * u_xx
    loss_f = mse_loss(res_u, torch.zeros_like(res_u)) + mse_loss(res_v, torch.zeros_like(res_v))
    
    # --- Boundary Loss ---
    pred_b = model(X_b)
    u_b = pred_b[:, 0:1]
    v_b = pred_b[:, 1:2]
    loss_b = mse_loss(u_b, torch.zeros_like(u_b)) + mse_loss(v_b, torch.zeros_like(v_b))
    
    # --- Initial Condition Loss ---
    pred_i = model(X_i)
    u_i_pred = pred_i[:, 0:1]
    v_i_pred = pred_i[:, 1:2]
    loss_i = mse_loss(u_i_pred, u_i) + mse_loss(v_i_pred, v_i)
    
    # Total loss
    loss = loss_f + loss_b + loss_i
    loss.backward()
    optimizer.step()
    
    # Store losses
    total_losses.append(loss.item())
    pde_losses.append(loss_f.item())
    bc_losses.append(loss_b.item())
    ic_losses.append(loss_i.item())
    
    # Update tqdm progress bar with losses in scientific notation
    pbar.set_postfix(
        total_loss=f"{loss.item():.4e}",
        pde_loss=f"{loss_f.item():.4e}",
        bc_loss=f"{loss_b.item():.4e}",
        ic_loss=f"{loss_i.item():.4e}"
    )

# ---------------------------
# Plotting the results: Loss Curves (log scale)
# ---------------------------
plt.figure(figsize=(8, 6))
epochs = range(num_epochs)
plt.semilogy(epochs, total_losses, label="Total Loss")
plt.semilogy(epochs, pde_losses, label="PDE Residual Loss")
plt.semilogy(epochs, bc_losses, label="Boundary Loss")
plt.semilogy(epochs, ic_losses, label="Initial Condition Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# ---------------------------
# Plotting the PINN solution and true solution for the real part
# ---------------------------
# Create a grid for plotting: x in [0,1] and t in [0,1]
N_x_plot = 100
N_t_plot = 100
x_plot = torch.linspace(x_min, x_max, N_x_plot)
t_plot = torch.linspace(t_min, t_max, N_t_plot)
X_grid, T_grid = torch.meshgrid(x_plot, t_plot, indexing='ij')
XT = torch.cat([X_grid.reshape(-1,1), T_grid.reshape(-1,1)], dim=1).to(device)

# Evaluate the PINN on the grid
with torch.no_grad():
    pred_plot = model(XT)
u_pred = pred_plot[:, 0].cpu().numpy().reshape(N_x_plot, N_t_plot)

# Compute the true solution for the real part.
# For the ground state, psi(x,t)= sin(pi*x)*exp(-i*pi^2*t/2), so:
# u_true(x,t)= sin(pi*x)*cos(pi^2*t/2)
X_np = X_grid.cpu().numpy()
T_np = T_grid.cpu().numpy()
u_true = np.sin(np.pi * X_np) * np.cos((np.pi**2/2)* T_np)

# Create subplots with time (t) on the vertical axis and x on the horizontal axis.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PINN prediction for the real part
c1 = axes[0].contourf(x_plot.cpu().numpy(), t_plot.cpu().numpy(), u_pred.T, 100, cmap='viridis')
axes[0].set_title("PINN Predicted Real Part")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t (time)")
fig.colorbar(c1, ax=axes[0])

# True solution for the real part
c2 = axes[1].contourf(x_plot.cpu().numpy(), t_plot.cpu().numpy(), u_true.T, 100, cmap='viridis')
axes[1].set_title("True Real Part")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t (time)")
fig.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.show()
