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

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return F.relu(self.linear(x))
    
class GELULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return F.gelu(self.linear(x))
    
class SiLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return F.silu(self.linear(x))
    
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

class WIRELayer(nn.Module):
    def __init__(self, in_features, out_features, s0, w0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("w0", torch.tensor(w0))
        self.register_buffer("s0", torch.tensor(s0))

    def forward(self, x):
        x = self.linear(x)
        x = torch.exp(-(self.s0 * x) ** 2) * torch.sin(self.w0 * x)
        return x


class MLPExpert(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        layernorm,
        layer_type="relu",
        omega_0=30,
        wire_s0=15.0,
        wire_w0=30.0,
    ):
        super().__init__()

        if n_layers < 2:
            raise ValueError("n_layers must be at least 2 to include a hidden and output layer")

        layer_type = layer_type.lower()

        def make_layer(in_features, out_features, is_first=False):
            if layer_type == "relu":
                return ReLULayer(in_features, out_features)
            elif layer_type in {"silu", "swish"}:
                return SiLULayer(in_features, out_features)
            elif layer_type == "gelu":
                return GELULayer(in_features, out_features)
            elif layer_type in {"sin", "sine"}:
                return SIRENLayer(
                    in_features,
                    out_features,
                    is_first=is_first,
                    omega_0=omega_0,
                )
            elif layer_type == "wire":
                return WIRELayer(
                    in_features=in_features,
                    out_features=out_features,
                    s0=wire_s0,
                    w0=wire_w0,
                )
            else:
                raise ValueError(f"Unknown layer_type {layer_type}")

        layers = []
        # first layer
        layers.append(make_layer(input_dim, hidden_dim, is_first=True))
        # hidden layers
        for _ in range(n_layers - 2):
            layers.append(make_layer(hidden_dim, hidden_dim))
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        
        if layernorm:
            self.layernorm = nn.LayerNorm(hidden_dim)
        else:
            self.layernorm = nn.Identity()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.layernorm(x)
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
                 freq_scale=40,
                 encoding_dim=256,
                 n_layers=5,
                 n_experts=4,
                 hidden_dim=512,
                 layernorm=True,
                 layer_type="relu"):
        super(GeneralModel, self).__init__()

        layer_type = layer_type.lower()
        disable_positional_encoding = layer_type in {"wire", "sin", "sine"}
        if disable_positional_encoding:
            position_encoding = None  # WIRE/SIREN use raw coordinates

        if position_encoding is None:
            self.pos_enc = nn.Identity()
            self.encoding_dim = input_dim
        elif position_encoding == "fourier":
            self.pos_enc = FourierFeatures(input_dim=input_dim,
                                           dim_per_input=encoding_dim//input_dim,
                                           trainable=trainable_embeddings,
                                           freq_scale=freq_scale)
            self.encoding_dim = encoding_dim
        elif position_encoding == "gabor":
            self.pos_enc = GaborFeatures(input_dim=input_dim,
                                         dim_per_input=encoding_dim//input_dim,
                                         trainable=trainable_embeddings,
                                         freq_scale=freq_scale)
            self.encoding_dim = encoding_dim

        self.output_dim = output_dim
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            MLPExpert(
                input_dim=self.encoding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                layernorm=layernorm,
                n_layers=n_layers,
                layer_type=layer_type,
            )
            for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, n_experts)
        )
        #self.gate = nn.Linear(input_dim, n_experts)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        gate_logits = self.gate(x)       # (B, n_experts)

        x = self.pos_enc(x)

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

    def compute_gate_weights(self, x):
        gate_logits = self.gate(x)
        return F.softmax(gate_logits, dim=-1)

