import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class GeneralModel(nn.Module):
    """
    Idea is to encode all relevant models in one
    general model and trigger behaviors by hyperparams
    which we can do a good search over
    """
    def __init__(self, input_dim=2, output_dim=3, position_encoding=None, encoding_dim=32, n_layers=3, hidden_dim=128, activation="relu"):
        super().__init__()

        if position_encoding is None:
            self.pos_enc = nn.Identity()
            self.input_dim = input_dim
        elif position_encoding is "fourier":
            self.pos_enc = FourierFeatures(encoding_dim=encoding_dim)
            self.input_dim = encoding_dim
        elif position_encoding is "gabor":
            self.pos_enc = GaborFeatures(encoding_dim=encoding_dim)
            self.input_dim = encoding_dim

        if activation == "relu":
            self.act = F.relu
        elif activation == "silu":
            self.act = F.silu
        elif activation == "sin":
            self.act = torch.sin

        self.layers = []
        current_dim = input_dim
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.pos_enc(x)

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.act(x)

        x = self.layers[-1](x)
        x = self.sigmoid(x)
        
        return x

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0,
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class SIREN(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=3, omega_0=60):
        super().__init__()
        self.net = []

        self.net.append(SIRENLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))

        for _ in range(hidden_layers):
            self.net.append(SIRENLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))

        self.net.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class GaborFeatures(nn.Module):
    def __init__(self, num_features=20, scale=40):
        super().__init__()
        
        self.sigmas = nn.Parameter(torch.randn(num_features)*scale)

    def forward(self, x):
        #input has shape [B, 2]

        #do the same calculation without list appending for efficiency
        sigmas = self.sigmas.view(1, -1)  # shape [1, 20]
        x = x.unsqueeze(-1)  # shape [B, 2, 1]
        x = torch.sin(x * sigmas)  # shape [B, 2, 20]
        x = x.view(x.size(0), -1)  # shape [B, 40]

        return x

class FourierFeatures(nn.Module):
    def __init__(self, num_features=20, scale=40):
        super().__init__()
        
        self.sigmas = nn.Parameter(torch.randn(num_features)*scale)

    def forward(self, x):
        #input has shape [B, 2]

        #do the same calculation without list appending for efficiency
        sigmas = self.sigmas.view(1, -1)  # shape [1, 20]
        x = x.unsqueeze(-1)  # shape [B, 2, 1]
        x = torch.sin(x * sigmas)  # shape [B, 2, 20]
        x = x.view(x.size(0), -1)  # shape [B, 40]

        return x

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super().__init__()
        self.features = FourierFeatures()

        self.net = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.net(x)

class MOEMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_experts=4):
        super().__init__()
        self.features = FourierFeatures()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
                nn.Sigmoid(),
            ) for _ in range(num_experts)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(40, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x_feat = self.features(x)  # shape [B, 40]
        gating_weights = self.gating_network(x_feat)  # shape [B, num_experts]

        expert_outputs = torch.stack([expert(x_feat) for expert in self.experts], dim=1)  # shape [B, num_experts, 3]

        # Weighted sum of expert outputs
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)  # shape [B, 3]

        return output

