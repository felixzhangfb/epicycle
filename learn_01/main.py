import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Epicycle:
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
            theta = -2 * np.pi / frames * frame
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

    @classmethod
    def animate_circles(
        cls, radius: list[float], theta: list[float], speed: list[float], orig: complex = 0 + 0j, frames: int = 360
    ):
        print(f'{radius=}')
        print(f'{theta=}')
        print(f'{speed=}')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_axis_off()
        xlim = sum(radius) * 1.1
        ylim = sum(radius) * 1.1
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        circles = []
        lines = []
        trace_points = []
        (trace,) = ax.plot([], [], color='red', lw=1)

        def init_func():
            center = orig
            t0 = 0
            for r, t in zip(radius, theta, strict=True):
                (circle,) = ax.plot([], [], color='#cccccc80', lw=2)
                t0 += t
                (line,) = ax.plot([], [], color='steelblue', lw=1)
                circles.append(circle)
                lines.append(line)
                center += r * np.exp(1j * t0)
            trace_points.clear()
            trace_points.append(center)
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circles + lines + [trace]

        def update(frame):
            center = orig
            t0 = 0
            for i, (r, t, s) in enumerate(zip(radius, theta, speed, strict=True)):
                circles[i].set_data(*cls.get_circle_points(center, r))
                t0 += t + np.pi / 180 * s * frame
                lines[i].set_data(*cls.get_arrow_points(center, r, t0))
                center += r * np.exp(1j * t0)
            # if len(trace_points) > 2 and np.abs(center - trace_points[0]) < 6.2e-15:
            #     raise ValueError('Circle center is already in trace_points')
            trace_points.append(center)
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circles + lines + [trace]

        _ = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init_func,
            interval=0.1,
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
    K = 3
    epicycle.animate_circles(
        radius=sorted(np.random.randint(1, 10, K), reverse=True),
        theta=list(np.random.randint(1, 10, K) / 10),
        speed=sorted(np.random.randint(-10, 10, K) / 10),
        orig=0 + 0j,
        frames=10000,
    )
