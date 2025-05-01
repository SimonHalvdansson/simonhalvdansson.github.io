import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------
# Define the PINN model for the ODE: y' = y
# ---------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=128, num_layers=3, activation=nn.Tanh()):
        """
        A configurable MLP for solving the ODE y' = y.
        - input_dim: dimension of input (t)
        - output_dim: dimension of output (y)
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
# Device configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Domain settings: t in [0,1]
# ---------------------------
N_f = 100  # Number of collocation points in the interior
t_f = torch.rand(N_f, 1, device=device)  # Uniformly sampled in [0,1]
t_f.requires_grad = True

# ---------------------------
# Initial condition: y(0)=1
# ---------------------------
t_b = torch.zeros(1, 1, device=device)  # t = 0
u_b = torch.ones_like(t_b)               # y(0) = 1

# ---------------------------
# Instantiate the model, optimizer, and loss function
# ---------------------------
hidden_dim = 128
num_layers = 3
activation = nn.Tanh()
model = PINN(input_dim=1, output_dim=1, hidden_dim=hidden_dim,
             num_layers=num_layers, activation=activation).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

# ---------------------------
# Training loop: enforce the ODE and the initial condition
# ---------------------------
num_epochs = 500
total_losses = []
pde_losses = []
bc_losses = []

pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    optimizer.zero_grad()
    
    # --- ODE Residual Loss ---
    # Enforce y' - y = 0 for t in (0,1)
    u_f_pred = model(t_f)  # Prediction at collocation points
    grad_u = torch.autograd.grad(u_f_pred, t_f, 
                                 grad_outputs=torch.ones_like(u_f_pred), 
                                 create_graph=True)[0]
    res = grad_u - u_f_pred  # Residual of the ODE y' - y = 0
    loss_f = mse_loss(res, torch.zeros_like(res))
    
    # --- Initial Condition Loss ---
    # Enforce y(0)=1
    u_b_pred = model(t_b)
    loss_b = mse_loss(u_b_pred, u_b)
    
    # Total loss
    loss = loss_f + loss_b
    loss.backward()
    optimizer.step()
    
    total_losses.append(loss.item())
    pde_losses.append(loss_f.item())
    bc_losses.append(loss_b.item())
    
    pbar.set_postfix(
        total_loss=f"{loss.item():.4e}",
        ode_loss=f"{loss_f.item():.4e}",
        ic_loss=f"{loss_b.item():.4e}"
    )

# ---------------------------
# Plotting the loss curves (log scale)
# ---------------------------
plt.figure(figsize=(8, 6))
epochs = range(num_epochs)
plt.semilogy(epochs, total_losses, label="Total Loss")
plt.semilogy(epochs, pde_losses, label="ODE Residual Loss")
plt.semilogy(epochs, bc_losses, label="Initial Condition Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# ---------------------------
# Plotting the PINN solution vs. the exact solution exp(t)
# ---------------------------
N_plot = 200
t_plot = np.linspace(0, 1, N_plot).reshape(-1, 1)
t_plot_tensor = torch.tensor(t_plot, dtype=torch.float32).to(device)
with torch.no_grad():
    u_pred = model(t_plot_tensor).cpu().numpy()
u_exact = np.exp(t_plot)

plt.figure(figsize=(8, 6))
plt.plot(t_plot, u_pred, label="PINN Prediction")
plt.plot(t_plot, u_exact, '--', label="Exact Solution: exp(t)")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Solution of ODE: y' = y, y(0)=1")
plt.legend()
plt.show()
