import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from flame5.variations import Variation


class SimpleTransform(nn.Module):
    def __init__(self, linear_transform=None):
        super().__init__()
        self.linear = torch.tensor(linear_transform, dtype=torch.float32) if linear_transform is not None else \
            torch.distributions.Normal(0, 2).sample((3, 2))
        self.linear = F.pad(self.linear, (0, 3 - self.linear.shape[1], 0, 3 - self.linear.shape[0]), 'constant', 0.0)
        self.linear[2, 2] = 1.0
        (self.a, self.b, self.c), (self.d, self.e, self.f), _ = torch.t(self.linear).detach().numpy()
        self.linear = self.linear.to('cuda')

    def forward(self, x):
        """

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor
        """
        return x @ self.linear


class RotationTransform(SimpleTransform):
    def __init__(self, angle):
        super().__init__(np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]))


class DihedralTransform(SimpleTransform):
    def __init__(self):
        super().__init__(np.array([
            [-1, 0],
            [ 0, 1]
        ]))


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = SimpleTransform().to('cuda')
        self.nonlinear_weights = F.softmax(torch.distributions.Uniform(0, 1).sample([len(Variation.all_variations)]))
        self.variation_chooser = torch.distributions.Multinomial(len(Variation.all_variations), self.nonlinear_weights)
        self.variations = nn.ModuleList([variation(self.linear, torch.distributions.Uniform(0, 1).sample([variation.num_p])).to('cuda')
                                         for variation in Variation.all_variations.values()])
        self.color = F.softmax(torch.distributions.Uniform(0, 1).sample([3]))
        self.post_transform = SimpleTransform().to('cuda')

    def forward(self, points):
        points = self.linear(points)
        func_choices = self.variation_chooser.sample([points.shape[0]]).to('cuda')
        for i, variation in enumerate(self.variations):
            selected = torch.nonzero(func_choices == i).to('cuda')
            points[selected, :] = variation(points[selected, :])
        points = self.post_transform(points)

        return points

    def plot(self, xmn=-1, xmx=1, xrs=41, ymn=-1, ymx=1, yrs=41):
        from matplotlib import pyplot as plt

        xs = torch.linspace(xmn, xmx, xrs)
        ys = torch.linspace(ymn, ymx, yrs)
        ones = torch.tensor([1], dtype=torch.float32)

        plt.figure(figsize=(6, 6))
        for y in ys:
            pts = torch.stack([xs, y.expand(xrs), ones.expand(xrs)], dim=1)
            pts = self(pts).detach().numpy()
            plt.plot(pts[:, 0], pts[:, 1], c='k')
        for x in xs:
            pts = torch.stack([x.expand(yrs), ys, ones.expand(yrs)], dim=1)
            pts = self(pts).detach().numpy()
            plt.plot(pts[:, 0], pts[:, 1], c='k')
        plt.show()


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    torch.manual_seed(1)

    transforms = torch.nn.ModuleList([Transform().to('cuda') for _ in range(3)])

    histogram = np.zeros((512, 512), dtype='float32')
    xbins = np.linspace(-1, 1, 513)
    ybins = np.linspace(-1, 1, 513)

    n_points = 1000000

    for _ in tqdm(range(10)):
        points = torch.cat([torch.randn((n_points, 2)), torch.ones((n_points, 1))], dim=1).to('cuda')

        for _ in tqdm(range(100), leave=False):
            t = np.random.choice(transforms)
            points = t(points)
            x, y, _ = points.to('cpu').detach().numpy().T
            histogram += np.histogram2d(x, y, bins=(xbins, ybins))[0]

        plt.figure()
        plt.imshow(np.log1p(histogram))
        plt.show(False)
