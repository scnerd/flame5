import itertools
import pickle
import random
from hashlib import md5
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .render import Renderer
from .runner import FunctionSet
from tqdm import tqdm
import argparse

def save_flame(args, flame):
    data = BytesIO()
    torch.save(flame, data)
    data = data.getvalue()

    args_data = pickle.dumps(args)

    hsh = md5(data).hexdigest()
    folder = Path('flames') / str(args.seed) / hsh

    folder.mkdir(parents=True, exist_ok=True)
    open(folder / 'function.tch', 'wb').write(data)
    open(folder / 'args.pkl', 'wb').write(args_data)

    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1000000, type=int)
    parser.add_argument('-s', default=500, type=int)
    parser.add_argument('--skip', default=100, type=int)
    parser.add_argument('-k', default=0, type=int)
    parser.add_argument('--base-width', default=2**13, type=int)
    parser.add_argument('--base-height', default=2**13, type=int)
    parser.add_argument('-z', default=5.0, type=float)
    parser.add_argument('-i', default=5, type=int)
    parser.add_argument('-j', default=5, type=int)
    parser.add_argument('--width', default=2**13, type=int)
    parser.add_argument('--height', default=2**13, type=int)
    parser.add_argument('--cmap', default='hsv', type=str)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--render-on-cpu', action='store_true', default=False)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    print(f'SEED: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    flame = FunctionSet.random_set(args.i, args.j)
    output_folder = save_flame(args, flame)
    renderer = Renderer(
        function_set=flame,
        width=args.base_width,
        height=args.base_height,
        x_left=-args.z,
        x_right=args.z,
        y_top=-args.z,
        y_bottom=args.z,
        palette=plt.get_cmap(args.cmap),
        palette_fidelity=100,
        gamma=args.gamma,
    )
    renderer = renderer.to('cuda')

    import threading
    import time

    keep_going = True
    def process():
        global keep_going
        for _ in (range(args.k) if args.k else itertools.count()):
            if not keep_going:
                break
            renderer(n_points=args.n, k_steps=args.s, skip_first_k=args.skip, progress=lambda a: tqdm(a, leave=False))
        keep_going = False


    def display():
        while keep_going:
            img = renderer.pretty_image(with_alpha=False, output_shape=(args.height, args.width), on_cpu=args.render_on_cpu)
            plt.imsave(f'flames/{args.seed}/render.png', img)
            time.sleep(5)

    process_thread = threading.Thread(target=process, daemon=True)
    display_thread = threading.Thread(target=display, daemon=True)
    try:
        process_thread.start()
        display_thread.start()
        while keep_going:
            time.sleep(1)
    except:
        print("Kill signal received, renderer terminating")
        keep_going = False
        process_thread.join(timeout=0)
        display_thread.join(timeout=0)

    img = renderer.pretty_image(with_alpha=False, output_shape=(args.base_height, args.base_width), on_cpu=True)
    plt.imsave(f'flames/{args.seed}/render.png', img)

    # for _ in tqdm(range(args.k)):
    #     renderer(n_points=args.n, k_steps=args.s, skip_first_k=args.skip, progress=lambda a: tqdm(a, leave=False))
    #     plt.imsave(output_folder / 'render.png', renderer.pretty_image(with_alpha=False, output_shape=(args.height, args.width)))
