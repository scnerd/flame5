# Overview

``flame5`` is a Pytorch implementation of the [Fractal Flames algorithm](https://flam3.com/flame_draves.pdf).

I originally tried implementing this algorithm in Theano, Tensorflow (TF 1.x), and raw Numba/numpy/cupy/etc. Theano and TF both build computation graphs (DAG's) that are easy to schedule and execute on CPU or GPU. However, part of the fractal flame algorithm requires that different data points have (randomly) different functions applied to them in an iterative fashion. Building this iteration and stochastic application of functions is very difficult in Theano and TF, and resulted in pretty brittle, hard-to-maintain code. Building my own CPU+GPU implementation using more basic libraries, using Numpy and Numba for CPU operations and Numba and Cupy to write GPU kernels, likewise got very complicated very fast, and was highly inflexible.

I chose Pytorch because its "just do what you want with whatever intermediate data you want and I'll schedule it when you're ready" approach is much more apt for this problem. I realize that TF now has an Eager mode that is pretty equivalent, but I find Pytorch easier anyway. That said, you can find my previous implementations in their own repositories.

# Installation

Install the ``requirements.txt`` dependencies. Some of these are easier to install with Conda (especially Pytorch with its CUDA dependency).

# Running

Just run the following:

```python -m flame5```

You can get help with ``-h`` to see the various CLI arguments available.

# TODO

- Implement all variations listed in the original paper

- Implement symmetry

- Implement vibrancy

- Add a GUI to more easily construct the flame

- Add animation

    - Add motion blur (keeping multiple linear accumulators for different points in time)
    
    - Add directional motion blur

- Add post-transforms

- Add final transform

- Implement piecewise rendering (render individual tiles of an image, then stitch them together to get ultra-high resolution)

- Support transform-dependent variations
