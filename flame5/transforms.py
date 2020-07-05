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
        self.linear = F.pad(self.linear, [0, 3 - self.linear.shape[1], 0, 3 - self.linear.shape[0]], 'constant', 0.0)
        self.linear[2, 2] = 1.0
        self.linear = nn.Parameter(self.linear, requires_grad=False)
        (self.a, self.b, self.c), (self.d, self.e, self.f), _ = torch.t(self.linear).detach().numpy()

    @torch.no_grad()
    def forward(self, x):
        """

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor
        """
        padded_x = F.pad(x, [0, 1], 'constant', 1.0)
        return (padded_x @ self.linear)[:, :2]

    def __getstate__(self):
        return dict(
            linear=self.linear.detach().cpu().numpy().tolist()
        )

    def __setstate__(self, state):
        self.linear = nn.Parameter(torch.tensor(state['linear'], dtype=torch.float32), requires_grad=False)
        (self.a, self.b, self.c), (self.d, self.e, self.f), _ = torch.t(self.linear).detach().numpy()


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


# class TransformBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = SimpleTransform().to('cuda')
#         self.nonlinear_weights = F.softmax(torch.distributions.Uniform(0, 1).sample([len(Variation.all_variations)]))
#         self.variation_chooser = torch.distributions.Multinomial(len(Variation.all_variations), self.nonlinear_weights)
#         self.variations = nn.ModuleList([
#             variation(torch.distributions.Uniform(0, 1).sample([variation.num_p])).to('cuda')
#             for variation in Variation.all_variations.values()
#         ])
#         self.color = F.softmax(torch.distributions.Uniform(0, 1).sample([3]))
#         self.post_transform = SimpleTransform().to('cuda')
#
#     @torch.no_grad()
#     def forward(self, points):
#         points = self.linear(points)
#         func_choices = self.variation_chooser.sample([points.shape[0]]).to('cuda')
#         for i, variation in enumerate(self.variations):
#             selected = torch.nonzero(func_choices == i).to('cuda')
#             points[selected, :] = variation(points[selected, :])
#         points = self.post_transform(points)
#
#         return points


if __name__ == '__main__':
    from . import grid_plot
    import torch

    torch.manual_seed(1)

    # transforms = torch.nn.ModuleList([SimpleTransform().to('cuda') for _ in range(3)])

    grid_plot(SimpleTransform().to('cuda'))

    # histogram = np.zeros((512, 512), dtype='float32')
    # xbins = np.linspace(-1, 1, 513)
    # ybins = np.linspace(-1, 1, 513)
    #
    # n_points = 1000000
    #
    # for _ in tqdm(range(10)):
    #     points = torch.cat([torch.randn((n_points, 2)), torch.ones((n_points, 1))], dim=1).to('cuda')
    #
    #     for _ in tqdm(range(100), leave=False):
    #         t = np.random.choice(transforms)
    #         points = t(points)
    #         x, y, _ = points.to('cpu').detach().numpy().T
    #         histogram += np.histogram2d(x, y, bins=(xbins, ybins))[0]
    #
    #     plt.figure()
    #     plt.imshow(np.log1p(histogram))
    #     plt.show(block=True)
