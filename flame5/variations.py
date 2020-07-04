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
        self.p = p

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

    @torch.no_grad()
    def forward(self, points: torch.Tensor):
        return self._forward(points)


class Linear(Variation):
    def _forward(self, points):
        return points


class Sinusoidal(Variation):
    def _forward(self, points):
        return torch.sin(points)


class Spherical(Variation):
    def _forward(self, points):
        return points / self.r(points).unsqueeze(1)


class Swirl(Variation):
    def _forward(self, points):
        x = self.x(points)
        y = self.y(points)
        r2 = self.r2(x, y)
        sin_r2 = torch.sin(r2)
        cos_r2 = torch.cos(r2)
        return torch.stack([
            x * sin_r2 - y * cos_r2,
            x * cos_r2 + y * sin_r2
        ], dim=1)


class Swirl2(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            r * torch.cos(theta + r),
            r * torch.sin(theta + r)
        ], dim=1)


class Horseshoe(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            r * torch.cos(2 * theta),
            r * torch.sin(2 * theta)
        ], dim=1)


class Polar(Variation):
    def _forward(self, points):
        return torch.stack([
            self.theta(points) / np.pi,
            self.r(points) - 1
        ], dim=1)


class Handkerchief(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            r * torch.sin(theta + r),
            r * torch.cos(theta - r)
        ], dim=1)


class Heart(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            r * torch.sin(theta * r),
            -r * torch.cos(theta * r)
        ], dim=1)


class Disc(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            theta * torch.sin(np.pi * r) / np.pi,
            theta * torch.cos(np.pi * r) / np.pi
        ], dim=1)


class Spiral(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            (torch.cos(theta) + torch.sin(r)) / r,
            (torch.sin(theta) - torch.cos(r)) / r
        ], dim=1)


class Hyperbolic(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            torch.sin(theta) / r,
            torch.cos(theta) * r
        ], dim=1)


class Diamond(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            torch.sin(theta) * torch.cos(r),
            torch.cos(theta) * torch.sin(r)
        ], dim=1)


class Ex(Variation):
    def _forward(self, points):
        r = self.r(points)
        theta = self.theta(points)
        return torch.stack([
            r * torch.pow(torch.sin(theta + r), 3),
            r * torch.pow(torch.cos(theta - r), 3)
        ], dim=1)


# class Julia(Variation):
#     num_p = 1
#
#     def _forward(self, points):
#         sqrt_r = torch.sqrt(self.r(points))
#         theta = self.theta(points)
#         return torch.stack([
#             sqrt_r * torch.cos(theta / 2 + self.p[0]),
#             sqrt_r * torch.sin(theta / 2 + self.p[0])
#         ], dim=1)


if __name__ == '__main__':
    from . import grid_plot

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('variation', choices=list(sorted(Variation.all_variations.keys())))
    args = parser.parse_args()

    variation = Variation.all_variations[args.variation]()
    grid_plot(variation.to('cuda', ))
