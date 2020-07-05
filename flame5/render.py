import numba as numba
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from .runner import FunctionSet


@numba.njit(parallel=True)
def _inverse_density_kde_filter(
    image,
    out,
    width,
    height,
    kde_width,
    kde_max_sigma,
):
    epsilon = 1.0 / kde_max_sigma

    for y in numba.prange(height):
        for x in range(width):
            density = image[3, y, x]
            sigma = 1.0 / (density + epsilon)
            sigma_squared = sigma ** 2.0
            for _y in range(-kde_width, kde_width + 1):
                for _x in range(-kde_width, kde_width + 1):
                    other_y = y + _y
                    other_x = x + _x
                    # other_density = image[3, other_y, other_x]
                    # https://mathworld.wolfram.com/BivariateNormalDistribution.html
                    # Correlation/covariance is 0
                    # so p = 0
                    weight = np.exp((_x ** 2 + _y ** 2) / sigma_squared) / (2.0 * np.pi * sigma_squared)
                    out[:, y, x] = image[:, other_y, other_x] * weight


class Renderer(nn.Module):
    def __init__(
        self,
        function_set: FunctionSet,
        height: int, width: int,
        x_left: float, x_right: float, y_top: float, y_bottom: float,
        palette = plt.get_cmap('viridis'),
        gamma: float = 2.2,
        initial_sigma: float = 2.0,
        palette_fidelity: int = 100
    ):
        super().__init__()

        self.function_set = function_set
        self.height = height
        self.width = width
        x_min = min(x_left, x_right)
        x_range = max(x_left, x_right) - x_min
        y_min = min(y_top, y_bottom)
        y_range = max(y_top, y_bottom) - y_min
        self.min_vec = nn.Parameter(torch.tensor([[x_min, y_min]], dtype=torch.float32), requires_grad=False)
        self.range_vec = nn.Parameter(
            torch.tensor([[width / x_range, height / y_range]], dtype=torch.float32),
            requires_grad=False
        )

        self.palette = nn.Parameter(torch.tensor([palette(v) for v in np.linspace(0, 1, palette_fidelity + 1)]))
        self.palette_fidelity = palette_fidelity
        self.gamma = gamma

        self.raw_image = nn.Parameter(torch.zeros((4, height, width)))
        self.initial_sigma = initial_sigma
        self.xy_generator = torch.distributions.Normal(loc=0.0, scale=initial_sigma)
        self.c_generator = torch.distributions.Uniform(0.0, 1.0)

    def _apply(self, *args, **kwargs):
        result = super()._apply(*args, **kwargs)
        result.xy_generator = torch.distributions.Normal(loc=torch.tensor(0.0).to(self.raw_image.device), scale=torch.tensor(result.initial_sigma).to(self.raw_image.device))
        result.c_generator = torch.distributions.Uniform(torch.tensor(0.0).to(self.raw_image.device), torch.tensor(1.0).to(self.raw_image.device))
        result.function_set = result.function_set._apply(*args, **kwargs)
        return result

    def convert_locations_to_bin_indices(self, points: torch.Tensor) -> torch.LongTensor:
        # noinspection PyTypeChecker
        return ((points - self.min_vec) * self.range_vec).long()

    @torch.no_grad()
    def forward(self, n_points, k_steps, skip_first_k=0, progress=lambda x: x):
        xy = self.xy_generator.sample((n_points, 2))
        c = self.c_generator.sample((n_points, 1))
        points = torch.cat([xy, c], dim=1)

        device = self.raw_image.device
        points = points.to(device)

        for k in progress(range(k_steps)):
            points = self.function_set(points)

            if k >= skip_first_k:
                xy, colors = points[:, :2], points[:, 2]
                # color_vectors = torch.tensor(self.palette(colors.detach().cpu().numpy()), dtype=torch.float32).to(device)
                # color_vectors = self.palette[(colors * (self.palette_fidelity - 1) + 0.5 / self.palette_fidelity).long()]
                _colors = colors * self.palette_fidelity
                _colors_floor = torch.floor(_colors)
                _colors_ceiling = torch.ceil(_colors)
                _colors_interp = (_colors - _colors_floor).unsqueeze(1)
                color_vectors = _colors_interp * self.palette[_colors_ceiling.long()] + (1 - _colors_interp) * self.palette[_colors_floor.long()]

                xy_bins = self.convert_locations_to_bin_indices(xy)
                x_bins, y_bins = xy_bins.T
                in_bounds = (x_bins >= 0) & (x_bins < self.width) & (y_bins >= 0) & (y_bins < self.height)
                x_bins = x_bins[in_bounds]
                y_bins = y_bins[in_bounds]
                color_vectors = color_vectors[in_bounds]
                for color, values in enumerate(color_vectors.T):
                    self.raw_image[color][[x_bins, y_bins]] += values

    def kde_blur(self, image, kde_width, kde_max_sigma):  # These should be class attributes...
        local_image = image.detach().cpu().numpy()
        filtered_image = np.zeros_like(local_image)
        _, height, width = image.shape
        _inverse_density_kde_filter(
            image=local_image,
            out=filtered_image,
            width=width,
            height=height,
            kde_width=kde_width,
            kde_max_sigma=kde_max_sigma
        )
        return torch.tensor(filtered_image, dtype=image.dtype, device=image.device)

    @torch.no_grad()
    def pretty_image(self, with_alpha=True, kde_width=0.05, kde_max_sigma=10, output_shape=None, on_cpu=False):
        if self.raw_image.max() == 0:
            if output_shape:
                shape = output_shape + (3,)
            else:
                shape = (self.height, self.width, 3)
            return np.zeros(shape)

        if on_cpu:
            img = self.raw_image.cpu()
        else:
            img = self.raw_image

        if output_shape:
            # Averaging results in a smooth result that goes through the nonlinear projections in the same way the original image would have, but with much less cost to render
            # In particular, the KDE filter is very expensive, and putting a smaller image through it saves a lot of time
            img = nn.AdaptiveAvgPool2d(output_shape)(img.unsqueeze(0))[0, ...]
        else:
            img = torch.clone(img)
        img += 1

        # Log-scale based on alpha
        alpha = img[3:, :, :]
        img = img * torch.log(alpha) / alpha

        # Density-inverse KDE bandwidth filtering
        # Blur each pixel by a gaussian whose std is inversely proportional to the
        #  density of information at a given pixel
        # Thus, darker, less-explored regions blur more, sharper details blur less
        img = self.kde_blur(img, kde_width, kde_max_sigma)

        # Gamma correction
        img = torch.pow(img, 1.0 / self.gamma)

        img /= img.max()
        if not with_alpha:
            img = img[:3, :, :]
        return img.permute((1, 2, 0)).detach().cpu().numpy()

    def heatmap(self):
        return self.raw_image[0, :, :].detach().cpu().numpy()

