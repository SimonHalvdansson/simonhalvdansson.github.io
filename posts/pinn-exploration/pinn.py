import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ---------------------------
# Define the PINN model for the harmonic function on a square
# ---------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3, activation=nn.Tanh()):
        """
        A configurable MLP for solving Laplace's equation on the square.
        - input_dim: dimension of input (x,y)
        - output_dim: dimension of output (u)
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
# Domain settings: square [-1,1] x [-1,1]
# ---------------------------
# Collocation points in the interior (for the PDE residual)
N_f = 10000
x_f = torch.rand(N_f, 1, device=device)*2 - 1  # Uniform in [-1,1]
y_f = torch.rand(N_f, 1, device=device)*2 - 1
X_f = torch.cat([x_f, y_f], dim=1)
X_f.requires_grad = True

# ---------------------------
# Define a parametrization gamma for the boundary
# ---------------------------
def gamma(t):
    """
    Parametrizes the boundary of the square [-1,1]x[-1,1] using a max-based method.
    For each t, computes
      x = cos(t)/max(|cos(t)|,|sin(t)|),
      y = sin(t)/max(|cos(t)|,|sin(t)|),
    so that (x,y) is on the square boundary.
    """
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    # Elementwise maximum to scale the point to the square boundary
    denom = torch.max(torch.abs(cos_t), torch.abs(sin_t))
    x = cos_t / denom
    y = sin_t / denom
    return x, y

# ---------------------------
# Boundary points: generate parameter t uniformly in [0, 2pi]
# ---------------------------
N_b = 200
t_b = torch.rand(N_b, 1, device=device)*2*np.pi
x_b, y_b = gamma(t_b)
X_b = torch.cat([x_b, y_b], dim=1)
# Boundary condition: u(gamma(t)) = sin(t)
u_b = torch.sin(t_b)

# ---------------------------
# Instantiate the model, optimizer, and loss function
# ---------------------------
hidden_dim = 128
num_layers = 3
activation = nn.Tanh()
model = PINN(input_dim=2, output_dim=1, hidden_dim=hidden_dim,
             num_layers=num_layers, activation=activation).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

# ---------------------------
# Training loop: enforce Laplace's equation and the boundary condition
# ---------------------------
num_epochs = 5000
total_losses = []
pde_losses = []
bc_losses = []

pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    optimizer.zero_grad()
    
    # --- PDE Residual Loss ---
    # Enforce u_xx + u_yy = 0 in the interior of the square
    u_f_pred = model(X_f)  # shape (N_f,1)
    grad_u = torch.autograd.grad(u_f_pred, X_f, 
                                 grad_outputs=torch.ones_like(u_f_pred), 
                                 create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X_f, 
                               grad_outputs=torch.ones_like(u_x), 
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X_f, 
                               grad_outputs=torch.ones_like(u_y), 
                               create_graph=True)[0][:, 1:2]
    res = u_xx + u_yy
    loss_f = mse_loss(res, torch.zeros_like(res))
    
    # --- Boundary Loss ---
    # At the boundary, u(gamma(t)) should equal sin(t)
    u_b_pred = model(X_b)
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
        pde_loss=f"{loss_f.item():.4e}",
        bc_loss=f"{loss_b.item():.4e}"
    )

# ---------------------------
# Plotting the loss curves (log scale)
# ---------------------------
plt.figure(figsize=(8, 6))
epochs = range(num_epochs)
plt.semilogy(epochs, total_losses, label="Total Loss")
plt.semilogy(epochs, pde_losses, label="PDE Residual Loss")
plt.semilogy(epochs, bc_losses, label="Boundary Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# ---------------------------
# Plotting the PINN solution over the square
# ---------------------------
N_grid = 200
x_plot = np.linspace(-1, 1, N_grid)
y_plot = np.linspace(-1, 1, N_grid)
X_grid, Y_grid = np.meshgrid(x_plot, y_plot)
XY = np.vstack((X_grid.flatten(), Y_grid.flatten())).T
XY_tensor = torch.tensor(XY, dtype=torch.float32).to(device)
with torch.no_grad():
    u_pred = model(XY_tensor).cpu().numpy().flatten()
u_pred_grid = u_pred.reshape(N_grid, N_grid)

plt.figure(figsize=(6,5))
cp = plt.contourf(x_plot, y_plot, u_pred_grid, 100, cmap='viridis')
plt.title("PINN Predicted Solution on the Square")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(cp)
plt.show()

# ---------------------------
# Plotting the boundary condition comparison
# ---------------------------
# Generate a dense set of t values for the boundary
t_plot = torch.linspace(0, 2*np.pi, 400, device=device).unsqueeze(1)
x_boundary, y_boundary = gamma(t_plot)
boundary_points = torch.cat([x_boundary, y_boundary], dim=1)
with torch.no_grad():
    u_boundary_pred = model(boundary_points).cpu().numpy().flatten()
u_boundary_true = torch.sin(t_plot).cpu().numpy().flatten()

plt.figure(figsize=(6,4))
plt.plot(t_plot.cpu().numpy(), u_boundary_pred, label='PINN Prediction')
plt.plot(t_plot.cpu().numpy(), u_boundary_true, '--', label='Boundary Condition (sin(t))')
plt.xlabel('t')
plt.ylabel('u')
plt.title("Boundary Condition Comparison")
plt.legend()
plt.show()
