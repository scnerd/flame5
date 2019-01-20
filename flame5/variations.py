import torch
from torch import nn


class Variation(nn.Module):
    all_variations = {}
    num_p = 0

    def __init__(self, transform, p=[]):
        super().__init__()
        self.t = transform
        self.p = p

    def __init_subclass__(cls, **kwargs):
        cls.all_variations[cls.__name__] = cls

    def forward(self, points):
        raise NotImplementedError()

    def plot(self, xmn=-1, xmx=1, xrs=41, ymn=-1, ymx=1, yrs=41):
        from matplotlib import pyplot as plt

        xs = torch.linspace(xmn, xmx, xrs)
        ys = torch.linspace(ymn, ymx, yrs)

        plt.figure(figsize=(6, 6))
        for y in ys:
            pts = torch.stack([xs, y.expand(xrs)], dim=1)
            pts = self(pts).detach().numpy()
            plt.plot(pts[:, 0], pts[:, 1], c='k')
        for x in xs:
            pts = torch.stack([x.expand(yrs), ys], dim=1)
            pts = self(pts).detach().numpy()
            plt.plot(pts[:, 0], pts[:, 1], c='k')
        plt.show()


class Linear(Variation):
    def forward(self, points):
        return points


class Sinusoidal(Variation):
    def forward(self, points):
        return torch.sin(points)


class Spherical(Variation):
    def forward(self, points):
        x = points[:, 0]
        y = points[:, 1]
        r = 1 / (x ** 2 + y ** 2)
        return r.unsqueeze(1) * points


class Swirl(Variation):
    def forward(self, points):
        x = points[:, 0]
        y = points[:, 1]
        r2 = x ** 2 + y ** 2
        return torch.stack([
            x * torch.sin(r2) - y * torch.cos(r2),
            x * torch.cos(r2) + y * torch.sin(r2)
        ], dim=1)
