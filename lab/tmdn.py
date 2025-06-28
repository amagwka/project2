import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


class GeometricAttention(nn.Module):
    """Attention mechanism that operates on geometric manifolds"""

    def __init__(self, dim: int, num_heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = dim // num_heads

        # Riemannian projections for manifold operations
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable manifold curvature for each head
        self.curvature = nn.Parameter(torch.randn(num_heads))

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """Compute hyperbolic distance with learnable curvature"""
        c = torch.abs(c) + 1e-8
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm = torch.sum(y * y, dim=-1, keepdim=True)

        # Poincar\xe9 ball distance
        num = 2 * torch.norm(x - y, dim=-1, keepdim=True) ** 2
        denom = (1 - c * x_norm) * (1 - c * y_norm)
        denom = torch.clamp(denom, min=1e-5)
        arg = 1 + c * num / denom
        arg = torch.clamp(arg, min=1.00001)
        return torch.acosh(arg) / torch.sqrt(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Compute attention weights using hyperbolic geometry
        attn_weights: List[torch.Tensor] = []
        for h in range(self.num_heads):
            c = self.curvature[h]
            q_h = q[:, :, h]
            k_h = k[:, :, h]

            distances = torch.zeros(B, T, T, device=x.device)
            for i in range(T):
                for j in range(T):
                    distances[:, i, j] = self.hyperbolic_distance(q_h[:, i], k_h[:, j], c).squeeze(-1)
            weights = F.softmax(-distances / self.temperature, dim=-1)
            attn_weights.append(weights)

        outputs = []
        for h, weights in enumerate(attn_weights):
            v_h = v[:, :, h]
            out_h = torch.bmm(weights, v_h)
            outputs.append(out_h)

        output = torch.cat(outputs, dim=-1)
        return self.out_proj(output)


class CausalConvolution(nn.Module):
    """Causal convolution with dilated temporal receptive field"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class WaveletTransform(nn.Module):
    """Learnable wavelet transform for multi-scale temporal analysis"""

    def __init__(self, dim: int, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        self.wavelets = nn.ParameterList([
            nn.Parameter(torch.randn(dim, 2 ** (i + 2))) for i in range(num_scales)
        ])
        self.reconstruct = nn.Linear(dim * num_scales, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_freq = x.transpose(1, 2)

        decompositions: List[torch.Tensor] = []
        for i, wavelet in enumerate(self.wavelets):
            wavelet_filter = F.softmax(wavelet, dim=-1)
            padding = wavelet_filter.size(-1) - 1
            x_padded = F.pad(x_freq, (padding, 0))
            decomp = F.conv1d(
                x_padded,
                wavelet_filter.unsqueeze(1),
                groups=self.dim,
            )
            stride = 2 ** (i + 1)
            decomp = decomp[:, :, ::stride]
            decomp = F.interpolate(decomp, size=T, mode='linear', align_corners=False)
            decompositions.append(decomp)

        combined = torch.cat(decompositions, dim=1).transpose(1, 2)
        return self.reconstruct(combined)


class TopologicalLayer(nn.Module):
    """Layer that preserves topological features across time"""

    def __init__(self, dim: int, persistence_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.persistence_dim = persistence_dim
        self.topology_encoder = nn.Sequential(
            nn.Linear(dim, persistence_dim),
            nn.ReLU(),
            nn.Linear(persistence_dim, persistence_dim)
        )
        self.topology_decoder = nn.Sequential(
            nn.Linear(persistence_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.birth_death = nn.Linear(persistence_dim, 2)

    def compute_persistence_loss(self, features: torch.Tensor) -> torch.Tensor:
        topo_features = self.topology_encoder(features)
        birth_death = self.birth_death(topo_features)
        births = birth_death[:, :, 0]
        deaths = birth_death[:, :, 1]
        persistence_loss = F.relu(births - deaths + 0.1).mean()
        return persistence_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topo_features = self.topology_encoder(x)
        topo_out = self.topology_decoder(topo_features)
        persistence_loss = self.compute_persistence_loss(x)
        return x + topo_out, persistence_loss


class TemporalManifoldDiffusionNetwork(nn.Module):
    """TMDN for video next-frame prediction"""

    def __init__(self, dinov2_dim: int = 384, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, sequence_length: int = 16, diffusion_steps: int = 10):
        super().__init__()
        self.dinov2_dim = dinov2_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.diffusion_steps = diffusion_steps
        self.input_proj = nn.Linear(dinov2_dim, hidden_dim)
        self.geometric_layers = nn.ModuleList([
            GeometricAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.wavelet_layers = nn.ModuleList([
            WaveletTransform(hidden_dim, num_scales=4) for _ in range(num_layers)
        ])
        self.topological_layers = nn.ModuleList([
            TopologicalLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.causal_convs = nn.ModuleList([
            CausalConvolution(hidden_dim, hidden_dim, 3, dilation=2 ** i) for i in range(num_layers)
        ])
        self.noise_scheduler = nn.Parameter(torch.linspace(0.1, 0.9, diffusion_steps))
        self.denoise_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, dinov2_dim)
        self.manifold_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def add_noise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        noise_level = self.noise_scheduler[t]
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, dinov2_embeddings: torch.Tensor, return_trajectory: bool = False):
        B, T, _ = dinov2_embeddings.shape
        x = self.input_proj(dinov2_embeddings)
        total_topo_loss = 0.0
        for geom_layer, wavelet_layer, topo_layer, causal_conv in zip(
            self.geometric_layers, self.wavelet_layers, self.topological_layers, self.causal_convs
        ):
            geom_out = geom_layer(x)
            wavelet_out = wavelet_layer(x)
            topo_out, topo_loss = topo_layer(x)
            total_topo_loss += topo_loss
            x_conv = x.transpose(1, 2)
            causal_out = causal_conv(x_conv).transpose(1, 2)
            x = self.layer_norm(geom_out + wavelet_out + topo_out + causal_out)
            x = self.dropout(x)
        last_frame = x[:, -1]
        if T > 1:
            penultimate_frame = x[:, -2]
            manifold_input = torch.cat([last_frame, penultimate_frame], dim=-1)
            interpolated = self.manifold_mixer(manifold_input)
            current_pred = last_frame + interpolated
        else:
            current_pred = last_frame
        trajectory = [current_pred] if return_trajectory else None
        for t in range(self.diffusion_steps):
            noisy_pred = self.add_noise(current_pred, t)
            t_emb = self.timestep_embedding(torch.full((B,), t, device=dinov2_embeddings.device), self.hidden_dim)
            denoiser_input = torch.cat([noisy_pred, t_emb], dim=-1)
            noise_pred = self.denoise_net(denoiser_input)
            current_pred = current_pred - noise_pred * self.noise_scheduler[t]
            if return_trajectory:
                trajectory.append(current_pred.clone())
        final_prediction = self.output_proj(current_pred)
        if return_trajectory:
            return final_prediction, trajectory, total_topo_loss
        return final_prediction, total_topo_loss


def create_tmdn_model() -> TemporalManifoldDiffusionNetwork:
    return TemporalManifoldDiffusionNetwork(
        dinov2_dim=384,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        sequence_length=16,
        diffusion_steps=10,
    )


def tmdn_loss(model_output: torch.Tensor, target: torch.Tensor, topological_loss: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    mse_loss = F.mse_loss(model_output, target)
    cosine_loss = 1 - F.cosine_similarity(model_output, target).mean()
    total_loss = mse_loss + cosine_loss + alpha * topological_loss
    return total_loss


def train_step(model: TemporalManifoldDiffusionNetwork, dinov2_sequence: torch.Tensor, next_frame_target: torch.Tensor):
    prediction, topo_loss = model(dinov2_sequence)
    loss = tmdn_loss(prediction, next_frame_target, topo_loss)
    return loss, prediction
