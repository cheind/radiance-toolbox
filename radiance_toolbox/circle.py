import numpy as np

from .proto import Shape, RAY_MAX
from matplotlib import patches


class Circle(Shape):
    def __init__(
        self,
        center: np.ndarray | None = None,
        radius: float | None = None,
    ):
        if center is None:
            center = np.zeros(2)
        if radius is None:
            radius = 1.0

        self.center = center
        self.radius = radius

    def intersect_closest(
        self,
        ray_o: np.ndarray,
        ray_d: np.ndarray,
    ) -> np.ndarray:

        v = self.center - ray_o
        t = np.tensordot(v, ray_d, (-1, -1))
        x = ray_o + t * ray_d
        xc = x - self.center
        d2 = np.tensordot(xc, xc, (-1, -1))
        mask = d2 <= self.radius**2
        k = np.sqrt(self.radius**2 - d2[mask])

        result = np.full(ray_o.shape[:-1], RAY_MAX, ray_d.dtype)
        result[mask] = t[mask] - k
        return result

    def normal(self, x: np.ndarray):
        v = x - self.center
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def plot(self, ax, **kwargs):
        defaults = {
            "linewidth": 1.0,
            "fill": False,
        }

        c = patches.Circle(xy=self.center, radius=self.radius, **{**defaults, **kwargs})
        ax.add_patch(c)
        return c


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    c = Circle(center=np.zeros(2), radius=0.5)
    ray_o = np.array([1.0, 1.0])
    ray_d = np.array([-1.0, -1.0])
    ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)

    print(ray_o, ray_d)

    r = c.intersect_closest(ray_o, ray_d)
    x = ray_o + r * ray_d
    n = c.normal(x)

    # valid = (r > ray_tmin) & (r < ray_tmax)

    print(r, x, n)

    fig, axs = plt.subplots()
    axs.set_aspect("equal")
    c.plot(axs)
    axs.autoscale_view()
    plt.show()
