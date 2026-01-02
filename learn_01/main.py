import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.animation import FuncAnimation


class Epicycle:
    def __init__(self):
        pass

    @staticmethod
    def get_circle_points(
        center: complex = 0 + 0j, radius: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = center + radius * np.exp(1j * theta)
        x = circle.real
        y = circle.imag
        return x, y

    @staticmethod
    def get_line_points(
        from_point: complex, to_point: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([from_point.real, to_point.real])
        y = np.array([from_point.imag, to_point.imag])
        return x, y

    @staticmethod
    def get_arrow_points(
        center: complex, radius: float, theta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        end_point = center + radius * np.exp(1j * np.pi * theta / 180)
        x = np.array([center.real, end_point.real])
        y = np.array([center.imag, end_point.imag])
        return x, y

    @classmethod
    def draw_circle(
        cls, center: complex = 0 + 0j, radius: float = 1.0, theta: float = 0.0
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.plot(*cls.get_circle_points(center, radius), color="#cccccc80", lw=5)
        ax.plot(*cls.get_arrow_points(center, radius, theta), color="steelblue", lw=2)
        plt.show()

    @classmethod
    def draw_circles(
        cls, radius: list[float], theta: list[float], orig: complex = 0 + 0j
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_axis_off()
        center = orig
        for r, t in zip(radius, theta):
            ax.plot(*cls.get_circle_points(center, r), color="#cccccc80", lw=5)
            ax.plot(*cls.get_arrow_points(center, r, t), color="steelblue", lw=2)
            center += r * np.exp(1j * np.pi * t / 180)
        plt.show()


def animate(i, epicycles, ax):
    for epicycle in epicycles:
        epicycle.step(0.01, 0)
    ax.clear()
    for epicycle in epicycles:
        epicycle.draw(ax)


if __name__ == "__main__":
    epicycle = Epicycle()
    epicycle.draw_circles(
        radius=[1.0, 0.5, 0.25], theta=[0.0, 30.0, -60.0], orig=0 + 0j
    )
