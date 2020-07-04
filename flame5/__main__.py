import matplotlib.pyplot as plt
import numpy as np
import torch

from .render import Renderer
from .runner import FunctionSet
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1000000, type=int)
    parser.add_argument('-k', default=1, type=int)
    parser.add_argument('--width', default=1024, type=int)
    parser.add_argument('--height', default=1024, type=int)
    parser.add_argument('--cmap', default='Accent', type=str)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    renderer = Renderer(
        function_set=FunctionSet.random_set(5, 5),
        width=args.width,
        height=args.height,
        x_left=-5,
        x_right=5,
        y_top=-5,
        y_bottom=5,
        palette=plt.get_cmap(args.cmap),
        palette_fidelity=10000,
        gamma=args.gamma,
    )
    renderer = renderer.to('cuda')

    for _ in tqdm(range(args.k)):
        renderer(n_points=args.n, k_steps=500, skip_first_k=100, progress=lambda a: tqdm(a, leave=False))
        plt.imsave('render.png', renderer.pretty_image(with_alpha=False))
