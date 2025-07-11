import torch
from config import Config
from scipy.integrate import solve_ivp
import numpy as np


class Interpolate:
    def __init__(self, config: Config):
        self.config = config
        self.b = config["interpolate"]["b"]
        self.schedule = config["interpolate"]["schedule"]
        self.c = config["sampling"]["c"]
        self.eps = self.b / 2.0
        # TODO: enable other schedules

        def w(t: float | np.ndarray, _: np.ndarray | None = None):
            if isinstance(t, np.ndarray):
                res = np.zeros_like(t)
                res[t < 1.0] = np.exp(-((1.0 - t[t < 1.0]) ** (-self.c)))
                return res
            elif isinstance(t, float):
                return np.exp(-((1.0 - t) ** (-self.c))) if t < 1.0 else 0.0
            else:
                raise ValueError()

        # set up time change
        theta_unscaled = solve_ivp(fun=w, t_span=(0, 1), y0=np.array(
            [0], dtype=np.float32), dense_output=True)

        def theta_t(t: float | np.ndarray | torch.Tensor):
            is_tensor = False
            if isinstance(t, torch.Tensor):
                is_tensor = True
                shape = t.shape
                dtype = t.dtype
                device = t.device
                t = t.numpy(force=True)
            res = w(t) / theta_unscaled.sol(1.0)
            return res if not is_tensor else torch.tensor(res, dtype=dtype, device=device).reshape(shape)

        def theta(t: float | np.ndarray | torch.Tensor):
            is_tensor = False
            if isinstance(t, torch.Tensor):
                is_tensor = True
                shape = t.shape
                dtype = t.dtype
                device = t.device
                t = t.numpy(force=True).flatten()
            res = theta_unscaled.sol(t) / theta_unscaled.sol(1.0)
            return res if not is_tensor else torch.tensor(res, dtype=dtype, device=device).reshape(shape).clamp(0.0, 1.0)

        self.theta_t = theta_t
        self.theta = theta

    def __call__(self, *args, **kwargs):
        return self.interpolate(*args, **kwargs)

    def interpolate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        """get xt from t, x0, x1, z
        """
        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        gamma = (self.b * t * (1.0 - t)) ** 0.5
        if self.schedule == "lerp":
            return (1.0 - t) * x0 + (t) * x1 + gamma * z
        elif self.schedule == "smoothstep":
            return (2 * t ** 3 - 3 * t ** 2 + 1) * x0 + (-2 * t ** 3 + 3 * t ** 2) * x1 + gamma * z
        else:
            raise ValueError()

    def get_target_EIt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        if self.schedule == "lerp":
            return x1 - x0
        elif self.schedule == "smoothstep":
            return (6 * t ** 2 - 6 * t) * x0 + (-6 * t ** 2 + 6 * t) * x1
        else:
            raise ValueError()

    def get_target_Ez(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        return z

    def get_weight_on_Ez(self, t: torch.Tensor, backward: bool):
        # note: this assumes that eps = b/2. by using closed-form, we avoid numerical error which was SIGNIFICANTLY degrading samples!

        res = torch.zeros_like(t)

        Q = 1e3

        # cutoff ensures that all items in res are less in magnitude than Q, for numerical stability
        if not backward:
            cutoff = (Q ** 2) / (self.b + Q ** 2)
            res[t < cutoff] = - \
                (((self.b * t[t < cutoff]) / (1.0 - t[t < cutoff])) ** 0.5)
            res[t >= cutoff] = -1e3
        else:
            cutoff = 1.0 - (Q ** 2) / (self.b + Q ** 2)
            res[t > cutoff] = - \
                (((self.b * (1.0 - t[t > cutoff])) / t[t > cutoff]) ** 0.5)
            res[t <= cutoff] = -1e3

        return res

    def get_target_forward_drift(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        """given a random sample from start and end distsm and time, calculate drift
        """

        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        drift = (self.get_target_EIt(t, x0, x1, z) +
                 self.get_weight_on_Ez(t, False) * z)

        return drift

    def get_target_backward_drift(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, z: torch.Tensor):
        assert t.dim() == 1
        assert x0.shape == x1.shape
        assert z.shape == x1.shape
        assert t.shape[0] == x0.shape[0]

        for _ in range(x0.dim() - t.dim()):
            t = t.unsqueeze(-1)

        drift = (-self.get_target_EIt(1.0 - t, x0, x1, z) + self.get_weight_on_Ez(1.0 -
                 t, True) * z)

        return drift
