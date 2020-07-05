import numpy as np
import torch
from torch import nn

from .variations import Variation, VariationSet
from .transforms import SimpleTransform


class Function(nn.Module):
    def __init__(self, transform: SimpleTransform, variation_set: VariationSet):
        super().__init__()

        self.transform = transform
        self.variation_set = variation_set

    def _apply(self, *args, **kwargs):
        result = super()._apply(*args, **kwargs)
        result.transform = result.transform._apply(*args, **kwargs)
        result.variation_set = result.variation_set._apply(*args, **kwargs)
        return result

    @torch.no_grad()
    def forward(self, points):
        return self.variation_set(self.transform(points))


class FunctionSet(nn.Module):
    def __init__(self, functions: [Function], probabilities: [float], colors: [torch.Tensor]):
        super().__init__()

        self.functions = nn.ModuleList(functions)
        probs = torch.tensor(probabilities, dtype=torch.float32)
        self.probabilities = nn.Parameter(probs / probs.sum(), requires_grad=False)
        self.chooser = torch.distributions.Categorical(probs=self.probabilities)
        self.colors = nn.Parameter(torch.tensor(colors, dtype=torch.float32))

    def _apply(self, *args, **kwargs):
        result = super()._apply(*args, **kwargs)
        # result.functions = nn.ModuleList([function.to(*args, **kwargs) for function in result.functions])
        result.chooser = torch.distributions.Categorical(probs=result.probabilities)
        return result

    @torch.no_grad()
    def forward(self, points):
        n, d = points.shape
        assert d == 3

        function_choices = self.chooser.sample([n])
        output_points = []
        for i, (function, function_color) in enumerate(zip(self.functions, self.colors)):
            selected_points = points[function_choices == i]
            xy, c = selected_points[:, :2], selected_points[:, 2:]
            assert c.shape[-1] in (3, 1)

            out_xy = function(xy)
            out_c = (c + function_color.view(1, -1)) / 2.0
            out = torch.cat([out_xy, out_c], dim=1)
            output_points.append(out)

        return torch.cat(output_points, dim=0)

    @classmethod
    def random_set(cls, num_functions, num_variations_per_function):
        functions = []
        for _ in range(num_functions):
            variations = [
                variation()
                for variation in np.random.choice(
                    list(Variation.all_variations.values()),
                    size=num_variations_per_function,
                    replace=False
                )
            ]
            variation_set = VariationSet(
                variations,
                np.random.uniform(size=num_variations_per_function)
            )
            function = Function(
                transform=SimpleTransform(),
                variation_set=variation_set,
            )
            functions.append(function)

        return cls(
            functions=functions,
            probabilities=np.random.uniform(size=num_functions),
            colors=np.random.uniform(size=num_functions)
        )


if __name__ == '__main__':
    from . import grid_plot
    import torch
    torch.manual_seed(1)

    from .variations import Swirl, Sinusoidal

    transform_1 = SimpleTransform()
    variation_set_1 = VariationSet(
        [Swirl(), Sinusoidal()],
        [0.5, 0.5]
    )
    function_1 = Function(transform=transform_1, variation_set=variation_set_1)
    grid_plot(function_1.to('cuda'))
