import torch
from config import Config


class Interpolate:
    def __init__(self, config: Config):
        self.config = config
        self.b = config["interpolate"]["b"]
        self.eps = self.b / 2.0
        # TODO: enable other schedules

    def __call__(self, *args, **kwargs):
        return self.interplate(*args, **kwargs)

    def interplate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        """get xt from t, x0, x1, z
        """
        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)
        gamma = (self.b * t * (1.0 - t)) ** 0.5
        return (1.0 - t) * x0 + (t) * x1 + gamma * z

    def get_target_EIt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        return x1 - x0

    def get_target_Ez(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        return z
    
    def get_weight_on_Ez(self, t: torch.Tensor, backward: bool):
        gamma_inv = (t * (1.0 - t)) ** (-0.5)
        
        if not backward:
            res: torch.Tensor = 0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv - self.eps * gamma_inv
        else:
            # XXX: must pass in 1-t
            res: torch.Tensor = -0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv - self.eps * gamma_inv
            
        res = res.clamp(-1e3, +1e3)
        
        return res
    
    def get_training_avr_weight_on_Ez(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        gamma_inv = (t * (1.0 - t)) ** (-0.5)
        
        forward_weight = (0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv - self.eps * gamma_inv) ** 2
        backward_weight = (0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv + self.eps * gamma_inv) ** 2
        
        weight = 0.5 * forward_weight + 0.5 * backward_weight
        return weight / (weight.sum() / weight.numel())
        

    def get_target_forward_drift(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        """given a random sample from start and end distsm and time, calculate drift
        """

        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        # TODO: support other interpolants
        # for now, we define
        # I(t, x0, x1) = (1-t) x0 + (t) x1
        # gamma(t) = sqrt(t(1-t))
        gamma_inv = (t * (1.0 - t)) ** (-0.5)

        velocity = x1 - x0 + 0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv * z
        score = - self.eps * gamma_inv * z

        return velocity + score

    def get_target_backward_drift(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        # TODO: support other interpolants
        # for now, we define
        # I(t, x0, x1) = (1-t) x0 + (t) x1
        # gamma(t) = sqrt(t(1-t))
        gamma_inv = (t * (1.0 - t)) ** (-0.5)

        velocity = -(x1 - x0 + 0.5 * ((self.b) ** 0.5) * (1.0 - 2.0 * t) * gamma_inv * z)
        score = -self.eps * gamma_inv * z

        return velocity + score
