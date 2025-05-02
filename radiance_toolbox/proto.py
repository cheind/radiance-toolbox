import numpy as np
from typing import Protocol
from matplotlib.axis import Axis

RAY_MIN = 0.0
RAY_MAX = np.finfo(np.float32).max


class Shape(Protocol):
    def intersect_closest(self, ray_o: np.ndarray, ray_d: np.ndarray) -> np.ndarray:
        pass

    def normal(self, x: np.ndarray) -> np.ndarray:
        pass

    def plot(self, ax: Axis):
        pass
