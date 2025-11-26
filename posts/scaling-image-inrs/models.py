import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class GaborFeatures(nn.Module):
    def __init__(self, input_dim, dim_per_input=20, freq_scale=40, trainable=False):
        super().__init__()
        
        centers = torch.rand(dim_per_input, input_dim)
        frequencies = torch.randn(dim_per_input, input_dim) * freq_scale
        sigmas = torch.full((dim_per_input, input_dim), 0.1)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.frequencies = nn.Parameter(frequencies)
            self.sigma = nn.Parameter(sigmas)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('frequencies', frequencies)
            self.register_buffer('sigma', sigmas)

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        envelope = torch.exp(- (diff ** 2 / (self.sigma.unsqueeze(0) ** 2)).sum(dim=2) / 2)
        phase = 2 * np.pi * (diff * self.frequencies.unsqueeze(0)).sum(dim=2)

        return torch.cat([envelope * torch.cos(phase), envelope * torch.sin(phase)], dim=-1)

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, dim_per_input=20, freq_scale=40, trainable=False):
        super().__init__()
        
        freqs = torch.randn(input_dim, dim_per_input) * freq_scale

        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def forward(self, x):
        phase = 2 * torch.pi * (x @ self.freqs)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation="relu"):
        super().__init__()

        if activation == "relu":
            self.act = F.relu
        elif activation == "silu":
            self.act = F.silu
        elif activation == "sin":
            self.act = torch.sin
        else:
            raise ValueError(f"Unknown activation {activation}")

        layers = []
        # first layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        # hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, input_dim)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x

class GeneralModel(nn.Module):
    """
    Idea is to encode all relevant models in one
    general model and trigger behaviors by hyperparams
    which we can do a good search over
    """
    def __init__(self, input_dim=2,
                 output_dim=3,
                 position_encoding="fourier",
                 trainable_embeddings = True,
                 encoding_dim=256,
                 n_layers=5,
                 n_experts=4,
                 hidden_dim=512,
                 activation="relu"):
        super(GeneralModel, self).__init__()

        if position_encoding is None:
            self.pos_enc = nn.Identity()
            self.encoding_dim = input_dim
        elif position_encoding == "fourier":
            self.pos_enc = FourierFeatures(input_dim=input_dim, dim_per_input=encoding_dim//input_dim, trainable=trainable_embeddings)
            self.encoding_dim = encoding_dim
        elif position_encoding == "gabor":
            self.pos_enc = GaborFeatures(input_dim=input_dim, dim_per_input=encoding_dim//input_dim, trainable=trainable_embeddings)
            self.encoding_dim = encoding_dim

        self.output_dim = output_dim
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            MLPExpert(
                input_dim=self.encoding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                activation=activation,
            )
            for _ in range(n_experts)
        ])

        self.gate = nn.Linear(self.encoding_dim, n_experts)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.pos_enc(x)

        gate_logits = self.gate(x)       # (B, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # mixture weights

        # experts
        expert_outputs = []
        for expert in self.experts:
            y = expert(x)                # (B, output_dim)
            expert_outputs.append(y.unsqueeze(-2))  # (B, 1, output_dim)

        # stack experts: (B, n_experts, output_dim)
        expert_outputs = torch.cat(expert_outputs, dim=-2)

        # mixture: sum_k w_k * y_k
        # gate_weights: (B, n_experts) -> (B, n_experts, 1)
        mixed = torch.sum(
            gate_weights.unsqueeze(-1) * expert_outputs,
            dim=-2
        )  # (B, output_dim)

        out = self.sigmoid(mixed)
        return out


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


class MOEMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_experts=4, hidden_dim=512):
        super().__init__()

        fourier_features = 60

        self.features = FourierFeatures(dim_per_input=fourier_features, input_dim=2)
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fourier_features*input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            ) for _ in range(num_experts)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(fourier_features*input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x_feat = self.features(x)  # shape [B, 40]
        gating_weights = self.gating_network(x_feat)  # shape [B, num_experts]

        expert_outputs = torch.stack([expert(x_feat) for expert in self.experts], dim=1)  # shape [B, num_experts, 3]

        # Weighted sum of expert outputs
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)  # shape [B, 3]

        return output

