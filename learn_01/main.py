import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Epicycle:
    def __init__(self):
        pass

    @staticmethod
    def get_circle_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0, 2 * np.pi, n_points)
        circle = center + radius * np.exp(1j * theta)
        x = circle.real
        y = circle.imag
        return x, y

    @staticmethod
    def get_arrow_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        theta: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        end_point = center + radius * np.exp(1j * theta)
        x = np.array([center.real, end_point.real])
        y = np.array([center.imag, end_point.imag])
        return x, y

    @classmethod
    def draw_circle(
        cls,
        center: complex = 0 + 0j,
        radius: float = 1.0,
        theta: float = 0.0,
        figsize: tuple[float, float] = (6, 6),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.plot(*cls.get_circle_points(center, radius), color='#cccccc80', lw=5)
        ax.plot(*cls.get_arrow_points(center, radius, theta), color='steelblue', lw=2)
        fig.show()

    @classmethod
    def draw_circles(
        cls,
        radius: list[float],
        theta: list[float],
        orig: complex = 0 + 0j,
    ):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.set_axis_off()
        center = orig
        for r, t in zip(radius, theta, strict=True):
            ax.plot(*cls.get_circle_points(center, r), color='#cccccc80', lw=5)
            ax.plot(*cls.get_arrow_points(center, r, t), color='steelblue', lw=2)
            center += r * np.exp(1j * t)
        fig.show()

    @classmethod
    def animate_circle(cls, center: complex, radius: float, speed: float, frames: int = 360):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
        (circle,) = ax.plot([], [], color='#cccccc80', lw=10)
        (line,) = ax.plot([], [], color='steelblue', lw=4)
        (trace,) = ax.plot([], [], color='red', lw=2)
        trace_points = []

        def init_func():
            circle.set_data(*cls.get_circle_points(center, radius, n_points=frames))
            line.set_data([], [])
            trace.set_data([], [])
            trace_points.clear()
            return circle, line, trace

        def update(frame):
            theta = - 2 * np.pi / frames * frame
            line.set_data(*cls.get_arrow_points(center, radius, theta))
            trace_points.append(center + radius * np.exp(1j * theta))
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circle, line, trace

        _ = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init_func,
            interval=speed,
            blit=True,
            repeat=False,
        )
        plt.tight_layout()
        plt.show()
        # plt.close(fig)
        # with open('animation.html', 'w') as f:
        #     f.write(ani.to_jshtml())


if __name__ == '__main__':
    epicycle = Epicycle()
    epicycle.animate_circle(center=0 + 0j, radius=2.0, speed=1, frames=500)
