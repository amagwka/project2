import torch


class BaseIntrinsicReward:
    """Interface for plug-in curiosity reward modules."""

    def reset(self) -> None:  # pragma: no cover - interface method
        """Reset any internal state maintained by the module."""
        raise NotImplementedError

    def compute(self, h):  # pragma: no cover - interface method
        """Return the intrinsic reward for the given embedding."""
        raise NotImplementedError

class E3BIntrinsicReward(BaseIntrinsicReward):
    def __init__(self, latent_dim=384, decay=0.9995, ridge=0.1, device="cpu"):
        """
        Args:
            latent_dim (int): The dimension of the embedding.
            decay (float): Exponential decay for soft resetting (0=no memory, 1=never resets).
            ridge (float): Regularization for the initial covariance inverse.
            device (str): PyTorch device.
        """
        self.latent_dim = latent_dim
        self.decay = decay
        self.ridge = ridge
        self.device = device
        self.reset()

    def reset(self):
        """Reset the inverse covariance matrix to its initial state."""
        self.cov_inverse = torch.eye(self.latent_dim, device=self.device) * (1.0 / self.ridge)

    def compute(self, h):
        """
        Compute E3B intrinsic reward for a given embedding and update covariance.
        Args:
            h (np.ndarray or torch.Tensor): The embedding vector, shape (latent_dim,).
        Returns:
            float: The intrinsic reward (Mahalanobis distance squared).
        """
        if not isinstance(h, torch.Tensor):
            h = torch.from_numpy(h).to(self.device)
        h = h.float()
        # Compute Mahalanobis-like novelty
        u = self.cov_inverse @ h
        novelty = torch.dot(h, u).item()
        # Update cov_inverse (Sherman-Morrison)
        u_unsq = u.unsqueeze(1)
        self.cov_inverse -= (u_unsq @ u_unsq.t()) / (1.0 + novelty)
        # Apply decay/soft reset
        self.cov_inverse = (
            self.decay * self.cov_inverse +
            (1 - self.decay) * torch.eye(self.latent_dim, device=self.device) * (1.0 / self.ridge)
        )
        return novelty

# Example usage:
if __name__ == "__main__":
    import numpy as np

    latent_dim = 384
    reward_module = E3BIntrinsicReward(latent_dim=latent_dim, decay=0.99, ridge=0.1, device="cpu")
    
    # Simulate streaming embeddings
    for step in range(5):
        fake_embedding = np.random.randn(latent_dim).astype(np.float32)
        reward = reward_module.compute(fake_embedding)
        print(f"Step {step}: intrinsic reward = {reward:.4f}")

    # Optionally reset
    reward_module.reset()
