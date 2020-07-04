import torch

torch.set_default_dtype(torch.float32)


@torch.no_grad()
def grid_plot(func, xmn=-1, xmx=1, xrs=41, ymn=-1, ymx=1, yrs=41, device=None):
    from matplotlib import pyplot as plt

    if device is None:
        try:
            device, = list(set(p.device for p in func.parameters()))
        except ValueError:
            device = 'cuda'

    xs = torch.linspace(xmn, xmx, xrs).to(device)
    ys = torch.linspace(ymn, ymx, yrs).to(device)
    ones = torch.tensor([1], dtype=torch.float32).to(device)

    plt.figure(figsize=(6, 6))
    for y in ys:
        pts = torch.stack([xs, y.expand(xrs), ones.expand(xrs)], dim=1)
        pts = func(pts).detach().cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], c='k')
    for x in xs:
        pts = torch.stack([x.expand(yrs), ys, ones.expand(yrs)], dim=1)
        pts = func(pts).detach().cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], c='k')
    plt.show()
