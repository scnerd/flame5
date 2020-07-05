import numpy as np
import torch
from torch import nn


class VariationSet(nn.Module):
    def __init__(self, variations, weights):
        super().__init__()
        self.variations = nn.ModuleList(variations)
        weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = nn.Parameter(weights / weights.sum(), requires_grad=False)

    @torch.no_grad()
    def forward(self, points):
        return torch.sum(
            torch.stack([variation(points) for variation in self.variations], dim=0) * self.weights.view(-1, 1, 1),
            dim=0
        )


class Variation(nn.Module):
    all_variations = {}
    num_p = 0

    def __init__(self, p=tuple()):
        super().__init__()
        # noinspection PyTypeChecker
        self.p = np.array(p).tolist()  # type: list

    def __init_subclass__(cls, **kwargs):
        cls.all_variations[cls.__name__] = cls

    @staticmethod
    def x(points: torch.Tensor) -> torch.Tensor:
        return points[:, 0]

    @staticmethod
    def y(points: torch.Tensor) -> torch.Tensor:
        return points[:, 1]

    @staticmethod
    def r(points_or_x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if y is not None:
            x = points_or_x
            return torch.sqrt(x ** 2 + y ** 2)
        else:
            points = points_or_x
            return torch.norm(points, dim=1)

    @staticmethod
    def r2(points_or_x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if y is not None:
            x = points_or_x
            return x ** 2 + y ** 2
        else:
            points = points_or_x
            return torch.pow(points, 2).sum(dim=1)

    @classmethod
    def theta(cls, points_or_x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if y is not None:
            x = points_or_x
        else:
            points = points_or_x
            x = cls.x(points)
            y = cls.y(points)
        return torch.atan2(y, x)

    @classmethod
    def phi(cls, points_or_x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if y is not None:
            x = points_or_x
        else:
            points = points_or_x
            x = cls.x(points)
            y = cls.y(points)
        return torch.atan2(x, y)

    @classmethod
    def omega(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.rand_like(x) >= 0.5, x.new_tensor(1), x.new_tensor(0))

    @classmethod
    def lambda_(cls, x):
        return torch.where(torch.rand_like(x) >= 0.5, x.new_tensor(1), x.new_tensor(-1))

    @classmethod
    def psi(cls, x):
        return torch.rand_like(x)

    @torch.no_grad()
    def forward(self, points: torch.Tensor):
        v = self._forward(points)
        if isinstance(v, tuple):
            v = torch.stack(list(v), dim=1)
        return v

    def __getstate__(self):
        return dict(
            p=self.p
        )


class Linear(Variation):
    def _forward(self, points):
        return points


class Sinusoidal(Variation):
    def _forward(self, points):
        return torch.sin(points)


class Spherical1(Variation):
    def _forward(self, points):
        return points / self.r(points).unsqueeze(1)


class Spherical2(Variation):
    def _forward(self, points):
        return points / self.r2(points).unsqueeze(1)


class Swirl(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r2 = self.r2(x, y)
        sin_r2 = torch.sin(r2)
        cos_r2 = torch.cos(r2)
        return (
            x * sin_r2 - y * cos_r2,
            x * cos_r2 + y * sin_r2
        )


class Swirl2(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            r * torch.cos(theta + r),
            r * torch.sin(theta + r)
        )


class Horseshoe(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            r * torch.cos(2 * theta),
            r * torch.sin(2 * theta)
        )


class Horseshoe2(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        return (
            (x - y) * (x + y) / r,
            2 * x * y / r
        )


class Polar(Variation):
    def _forward(self, points):
        return (
            self.theta(points) / np.pi,
            self.r(points) - 1
        )


class Handkerchief(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            r * torch.sin(theta + r),
            r * torch.cos(theta - r)
        )


class Heart(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            r * torch.sin(theta * r),
            -r * torch.cos(theta * r)
        )


class Spiral(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            (torch.cos(theta) + torch.sin(r)) / r,
            (torch.sin(theta) - torch.cos(r)) / r
        )


class Hyperbolic(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            torch.sin(theta) / r,
            torch.cos(theta) * r
        )


class Diamond(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            torch.sin(theta) * torch.cos(r),
            torch.cos(theta) * torch.sin(r)
        )


class Ex(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return (
            r * torch.pow(torch.sin(theta + r), 3),
            r * torch.pow(torch.cos(theta - r), 3)
        )


class Ex2(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        p0 = torch.sin(theta + r) ** 3
        p1 = torch.cos(theta - r) ** 3
        return (
            r * (p0 + p1),
            r * (p0 - p1)
        )


class Julia(Variation):
    def _forward(self, points):
        sqrt_r = torch.sqrt(self.r(points))
        theta = self.theta(points)
        omega = self.omega(sqrt_r)
        return (
            sqrt_r * torch.cos(theta / 2 + omega),
            sqrt_r * torch.sin(theta / 2 + omega)
        )
    
    
class Disc(Variation):
    def _forward(self, points):
        theta_pi = np.pi * self.theta(points)
        r_pi = np.pi * self.r(points)
        return (
            theta_pi * torch.sin(r_pi),
            theta_pi * torch.cos(r_pi)
        )
    
    
class Bent(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        return torch.where(x >= 0, x, 2 * x), torch.where(y >= 0, y, y / 2)


# class Waves(Variation):
#     pass


class Fisheye(Variation):
    def _forward(self, points):
        scale = 2 / (self.r(points) + 1)
        return (  # Yes, this ordering is intentional. See Eyefish for the "fixed" version...
            self.y(points) * scale,
            self.x(points) * scale
        )


# class Popcorn(Variation):
#     pass


class Exponential(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        scale = torch.exp(x - 1)
        return (
            scale * torch.cos(np.pi * y),
            scale * torch.sin(np.pi * y)
        )


class Power(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        theta = self.theta(x, y)
        scale = r ** torch.sin(theta)
        return (
            scale * torch.cos(theta),
            scale * torch.sin(theta)
        )


class Cosine(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        theta = self.theta(x, y)
        return (
            torch.cos(np.pi * x) * torch.cosh(y),
            -torch.sin(np.pi * x) * torch.sinh(y)
        )


# class Rings(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )


# class Fan(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )


class Blob(Variation):
    num_p = 3

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        theta = self.theta(x, y)
        high, low, waves = points.new_tensor(self.p)
        scale = r * (low + (high - low) / 2 * (torch.sin(waves * theta) + 1))
        return (
            scale * torch.cos(theta),
            scale * torch.sin(theta)
        )


class PDJ(Variation):
    num_p = 4

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        return (
            torch.sin(self.p[0] * y) - torch.cos(self.p[1] * x),
            torch.sin(self.p[2] * x) - torch.cos(self.p[3] * y)
        )


class Fan2(Variation):
    num_p = 2

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        theta = self.theta(x, y)
        j = np.pi * self.p[0] ** 2
        t = theta + self.p[1] - j * torch.trunc(2 * theta * self.p[1] / j)
        condition = t > j / 2
        theta_plus = theta + j / 2
        theta_minus = theta - j / 2
        return (
            r * torch.where(condition, torch.sin(theta_minus), torch.sin(theta_plus)),
            r * torch.where(condition, torch.cos(theta_minus), torch.cos(theta_plus))
        )


class Rings2(Variation):
    num_p = 1

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        theta = self.theta(x, y)
        p = self.p[0] ** 2
        t = r - 2 * p * torch.trunc((r + p) / (2 * p)) + r * (1 - p)
        return (
            t * torch.sin(theta),
            t * torch.cos(theta)
        )


class Eyefish(Variation):
    def _forward(self, points):
        scale = 2 / (self.r(points) + 1)
        return scale.unsqueeze(1) * points


class Bubble(Variation):
    def _forward(self, points):
        scale = 4 / (self.r2(points) + 4)
        return scale.unsqueeze(1) * points


class Cylinder(Variation):
    def _forward(self, points):
        return (
            torch.sin(self.x(points)),
            self.y(points)
        )


class Perspective(Variation):
    num_p = 2

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        angle, dist = points.new_tensor(self.p)
        scale = dist / (dist - y * torch.sin(angle))
        return (
            scale * x,
            scale * y * torch.cos(angle)
        )


class Noise(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r1 = self.psi(x)
        r2 = self.psi(x)
        return (
            r1 * x * torch.cos(2 * np.pi * r2),
            r1 * y * torch.sin(2 * np.pi * r2)
        )


class JuliaN(Variation):
    num_p = 2

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        power, dist = points.new_tensor(self.p)
        p3 = torch.trunc(torch.abs(power) * self.psi(x))
        t = (self.phi(points) + 2 * np.pi * p3) / power
        scale = r ** (dist / power)
        return (
            scale * torch.cos(t),
            scale * torch.sin(t)
        )


class JuliaScope(Variation):
    num_p = 2

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r = self.r(x, y)
        power, dist = points.new_tensor(self.p)
        p3 = torch.trunc(torch.abs(power) * self.psi(x))
        t = (self.lambda_(x) * self.phi(points) + 2 * np.pi * p3) / power
        scale = r ** (dist / power)
        return (
            scale * torch.cos(t),
            scale * torch.sin(t)
        )


class Blur(Variation):
    def _forward(self, points):
        x = self.x(points)
        r1 = self.psi(x)
        r2 = self.psi(x)
        return (
            r1 * torch.cos(2 * np.pi * r2),
            r1 * torch.sin(2 * np.pi * r2)
        )


class Gaussian(Variation):
    # I deviate slightly here; they do a gaussian approximation, I'll just use randn
    def _forward(self, points):
        x = self.x(points)
        r1 = torch.randn_like(x)
        r2 = self.psi(x)
        return (
            r1 * torch.cos(2 * np.pi * r2),
            r1 * torch.sin(2 * np.pi * r2)
        )


# class RadialBlur(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )


class Pie(Variation):
    num_p = 3

    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        t1 = torch.trunc(self.psi(x) * self.p[0] + 0.5)
        t2 = self.p[1] + 2 * np.pi / self.p[0] * (t1 + self.psi(x) * self.p[2])
        scale = self.psi(x)
        return (
            scale * torch.cos(t2),
            scale * torch.sin(t2)
        )


# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )
#
#
# class NAME(Variation):
#     def _forward(self, points):
#         x = self.x(points)
#         y = self.y(points)
#         r = self.r(x, y)
#         theta = self.theta(x, y)
#         return (
#
#         )


if __name__ == '__main__':
    from . import grid_plot

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('variation', choices=list(sorted(Variation.all_variations.keys())))
    args = parser.parse_args()

    variation = Variation.all_variations[args.variation]()
    grid_plot(variation.to('cuda', ))
