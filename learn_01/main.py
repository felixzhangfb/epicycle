import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray


class Epicycle:
    """A class for creating and animating epicycles (circular epitrochoids).

    An epicycle is a mathematical curve composed of multiple circles where each
    circle's center lies on the circumference of the previous circle. This class
    provides functionality to create, draw, and animate these geometric structures.

    Main features:
    - Generate coordinate points for circles
    - Draw static circles and circular chains
    - Create animations for single and multiple circles
    - Visualize epicycle motion trajectories

    Example:
        >>> epicycle = Epicycle()
        >>> epicycle.draw_circle(radius=2.0, theta=0.5)
        >>> epicycle.animate_circles([1, 2, 3], [0, 1, 2], [1, 0.5, 0.25])
    """

    @staticmethod
    def _is_jupyter():
        """Return True if running inside a Jupyter (ZMQ) kernel.

        Returns:
            bool: True if executed in a Jupyter notebook, otherwise False.
        """
        try:
            from IPython.core.getipython import get_ipython as _get_ipython

            ip = _get_ipython()
            if ip is None:
                return False
            return ip.__class__.__name__ == 'ZMQInteractiveShell'
        except Exception:
            return False

    @staticmethod
    def _get_circle_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        n_points: int = 100,
    ) -> tuple[NDArray, NDArray]:
        """Return x, y coordinate arrays sampled from a circle.

        Args:
            center (complex): Circle center. Defaults to 0+0j.
            radius (float): Circle radius. Defaults to 1.0.
            n_points (int): Number of points to sample. Defaults to 100.

        Returns:
            tuple[NDArray, NDArray]: x and y coordinate arrays.
        """
        theta = np.linspace(0, 2 * np.pi, n_points) 
        circle = center + radius * np.exp(1j * theta)
        x = circle.real
        y = circle.imag
        return x, y

    @staticmethod
    def _get_arrow_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        theta: float = 0.0,
    ) -> tuple[NDArray, NDArray]:
        """Return start/end coordinates of a radius line at angle ``theta``.

        Args:
            center (complex): Circle center.
            radius (float): Circle radius.
            theta (float): Angle in radians.

        Returns:
            tuple[NDArray, NDArray]: x and y arrays of length 2 (start, end).
        """
        end_point = center + radius * np.exp(1j * theta)
        x = np.array([center.real, end_point.real])
        y = np.array([center.imag, end_point.imag])
        return x, y

    @classmethod
    def _draw_circle(
        cls,
        center: complex = 0 + 0j,
        radius: float = 1.0,
        theta: float = 0.0,
        figsize: tuple[float, float] = (6, 6),
    ):
        """Draw a circle with a radius line using Matplotlib.

        Args:
            center (complex): Circle center.
            radius (float): Circle radius.
            theta (float): Angle for the radius line (radians).
            figsize (tuple): Figure size in inches.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.plot(*cls._get_circle_points(center, radius), color='#cccccc80', lw=5)
        ax.plot(*cls._get_arrow_points(center, radius, theta), color='tab:blue', lw=2)
        fig.show()

    @classmethod
    def _draw_circles(
        cls,
        radius: list[float],
        theta: list[float],
        orig: complex = 0 + 0j,
    ):
        """Draw multiple connected circles forming an epicycle chain.

        Args:
            radius (list[float]): Radii for each circle.
            theta (list[float]): Initial angles for each circle (radians).
            orig (complex): Starting origin for the first circle.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.set_axis_off()
        center = orig
        for r, t in zip(radius, theta, strict=True):
            ax.plot(*cls._get_circle_points(center, r), color='#cccccc80', lw=5)
            ax.plot(*cls._get_arrow_points(center, r, t), color='tab:blue', lw=2)
            center += r * np.exp(1j * t)
        fig.show()

    @classmethod
    def _animate_circle(
        cls,
        center: complex,
        radius: float,
        speed: float,
        frames: int = 360,
    ):
        """Animate a rotating circle and plot its trajectory.

        Args:
            center (complex): Circle center.
            radius (float): Circle radius.
            speed (float): Frame interval / speed control.
            frames (int): Number of frames (default 360).
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
        (circle,) = ax.plot([], [], color='tab:gray', lw=10, alpha=0.5)
        (line,) = ax.plot([], [], color='tab:blue', lw=4)
        (trace,) = ax.plot([], [], color='red', lw=2)
        trace_points = []

        def init_func():
            circle.set_data(*cls._get_circle_points(center, radius, n_points=frames))
            line.set_data([], [])
            trace.set_data([], [])
            trace_points.clear()
            return circle, line, trace

        def update(frame):
            theta = -2 * np.pi / frames * frame
            line.set_data(*cls._get_arrow_points(center, radius, theta))
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
        cls,
        radius: list[float] | None = None,
        theta: list[float] | None = None,
        speed: list[float] | None = None,
        orig: complex = 0 + 0j,
        n_frames: int = 10000,
        interval: float = 10,
        gif_name: str = 'circles_animation.gif',
    ):
        """Animate a chain of rotating circles and trace the final endpoint.

        Args:
            radius (list[float] | None): Radii for each circle. Defaults are used if None.
            theta (list[float] | None): Initial angles per circle (radians).
            speed (list[float] | None): Rotation speed per circle (radians/frame).
            orig (complex): Origin for the first circle.
            n_frames (int): Maximum number of frames.
            interval (float): Delay between frames (ms).
            gif_name (str): Filename to save GIF when running in Jupyter.
        """
        radius = radius or [2.0, 2.0, 1.0]
        theta = theta or [0.0, np.pi/4, np.pi/2]
        speed = speed or [0.01, 0.02, 0.03]

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
        (trace,) = ax.plot([], [], color='tab:red', lw=1)

        def init_func():
            center = orig
            for r, t in zip(radius, theta, strict=True):
                (circle,) = ax.plot([], [], color='tab:gray', ls='-', lw=1, alpha=0.5)
                (line,) = ax.plot([], [], color='tab:blue', lw=1)
                circles.append(circle)
                lines.append(line)
                # use the circle's own initial angle `t` (do not accumulate)
                center += r * np.exp(1j * t)
            trace_points.clear()
            trace_points.append(center)
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circles + lines + [trace]

        def frame_gen():
            for i in range(n_frames):
                if len(trace_points) > 2 and np.isclose(trace_points[-1], trace_points[0], atol=1e-6):
                    break
                yield i

        def update(frame):
            center = orig
            for i, (r, t, s) in enumerate(zip(radius, theta, speed, strict=True)):
                circles[i].set_data(*cls._get_circle_points(center, r))
                # compute angle for this frame without mutating the original `theta` list
                theta_i = t + s * frame
                lines[i].set_data(*cls._get_arrow_points(center, r, theta_i))
                center += r * np.exp(1j * theta_i)
            trace_points.append(center)
            trace.set_data(
                [p.real for p in trace_points],
                [p.imag for p in trace_points],
            )
            return circles + lines + [trace]

        ani = FuncAnimation(
            fig,
            update,
            frames=frame_gen(),
            init_func=init_func,
            interval=interval,
            blit=True,
            repeat=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        if cls._is_jupyter():
            from IPython.display import Image
            from IPython.display import display as _display

            ani.save(gif_name, writer=PillowWriter(fps=30))
            _display(Image(filename=gif_name))
        else:
            plt.show()

    @staticmethod
    def _generate_polygon_points(n_polygon_points: int = 20) -> NDArray[np.complex128]:
        """Generate normalized polygon points as complex numbers.

        Args:
            n_polygon_points (int): Number of polygon vertices to generate.

        Returns:
            NDArray[np.complex128]: Array of complex polygon points (closed, normalized).
        """
        x = np.random.randint(-10, 10, n_polygon_points)
        y = np.random.randint(-10, 10, n_polygon_points)
        z = x + 1j * y
        center = z.mean()
        z_shifted = z - center
        angles = np.angle(z_shifted)
        order = np.argsort(angles)
        z_order = z[order]
        z_order = np.r_[z_order, z_order[0]]
        z_order -= center
        z_order /= np.max(np.abs(z_order))
        return z_order

    @staticmethod
    def _resample_polygon_points(
        points: NDArray[np.complex128],
        n_sample_points: int = 50,
    ) -> NDArray[np.complex128]:
        """Linearly resample polygon points along the polygon perimeter.

        Args:
            points (NDArray): Complex polygon points (closed).
            n_sample_points (int): Number of output sample points.

        Returns:
            NDArray[np.complex128]: Resampled complex points.
        """
        diffs = np.diff(points)
        dists = np.abs(diffs)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_dist = cum_dists[-1]
        sample_dists = np.linspace(0, total_dist, n_sample_points)
        x_sample = np.interp(sample_dists, cum_dists, points.real)
        y_sample = np.interp(sample_dists, cum_dists, points.imag)
        sample_points = x_sample + 1j * y_sample
        return sample_points

    @staticmethod
    def _fft_sample_points(
        sample_points: NDArray[np.complex128],
        n_fft_points: int = 100,
    ) -> tuple[
        NDArray[np.complex128],
        list[float],
        list[float],
        list[float],
    ]:
        """Compute FFT modes and return reconstruction and parameters.

        Args:
            sample_points (NDArray): Complex-valued samples.
            n_fft_points (int): Number of FFT modes to retain.

        Returns:
            tuple: (fft_points: NDArray, radius: list, theta: list, speed: list)
        """
        n_sample_points = len(sample_points)
        freqs = np.fft.fftfreq(n_sample_points)
        fft_values = np.fft.fft(sample_points) / n_sample_points
        idx = np.argsort(np.abs(fft_values))[::-1]
        freqs = freqs[idx]
        fft_values = fft_values[idx]

        freqs = freqs[:n_fft_points]
        fft_values = fft_values[:n_fft_points]

        radius = np.abs(fft_values)
        theta = np.angle(fft_values)
        speed = 2 * np.pi * freqs

        fft_points = np.asarray([np.sum(fft_values * np.exp(1j * speed * t)) for t in range(n_sample_points)])
        fft_points = np.r_[fft_points, fft_points[0]]
        return fft_points, radius.tolist(), theta.tolist(), speed.tolist()

    @classmethod
    def animate_polygon_points(
        cls,
        n_polygon_points: int = 20,
        n_sample_points: int = 80,
        n_fft_points: int = 100,
        n_frames: int = 10000,
        interval: float = 20,
        gif_name: str = 'polygon_animation.gif',
    ):
        """Animate epicycle reconstruction of a random polygon via FFT modes.

        Args:
            n_polygon_points (int): Number of polygon vertices.
            n_sample_points (int): Resampled points along polygon.
            n_fft_points (int): Number of FFT modes to keep.
            n_frames (int): Maximum frames for the animation.
            interval (float): Delay between frames (ms).
            gif_name (str): GIF filename used when saving in Jupyter.
        """
        polygon_points = cls._generate_polygon_points(n_polygon_points)
        sample_points = cls._resample_polygon_points(polygon_points, n_sample_points=n_sample_points)
        fft_points, radius, theta, speed = cls._fft_sample_points(sample_points, n_fft_points=n_fft_points)

        fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        axes = axes.flatten()
        for ax in axes:
            ax.set_aspect('equal')
            # ax.set_axis_off()
        ax_polygon, ax_sample, ax_fft, ax_anim = axes
        ax_polygon.set_title('Polygon Points')
        ax_sample.set_title('Sample Points')
        ax_fft.set_title('FFT Points')
        ax_anim.set_title('Animation')

        ax_polygon.plot(
            polygon_points.real,
            polygon_points.imag,
            marker='o',
            color='tab:blue',
            lw=1,
            markersize=2,
        )
        xlim = ax_polygon.get_xlim()
        ylim = ax_polygon.get_ylim()

        ax_sample.plot(
            sample_points.real,
            sample_points.imag,
            marker='o',
            color='tab:orange',
            lw=1,
            markersize=2,
        )
        ax_fft.plot(
            fft_points.real,
            fft_points.imag,
            marker='o',
            color='tab:green',
            lw=1,
            markersize=2,
        )

        ax_anim.set_xlim(*xlim)
        ax_anim.set_ylim(*ylim)

        circles = []
        lines = []
        trace_points = []
        (trace,) = ax_anim.plot([], [], marker='o', color='tab:red', lw=1, markersize=2)

        def init_func():
            center = 0 + 0j
            for r, t in zip(radius, theta, strict=True):
                (circle,) = ax_anim.plot([], [], color='tab:gray', ls='-', lw=1, alpha=0.5)
                (line,) = ax_anim.plot([], [], color='tab:blue', lw=1)
                circles.append(circle)
                lines.append(line)
                center += r * np.exp(1j * t)
            trace_points.clear()
            trace_points.append(center)
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circles + lines + [trace]

        def frame_gen():
            for i in range(n_frames):
                if len(trace_points) > 2 and np.isclose(trace_points[-1], trace_points[0], atol=1e-10):
                    break
                yield i

        def update(frame):
            center = 0 + 0j
            for i, (r, t, s) in enumerate(zip(radius, theta, speed, strict=True)):
                circles[i].set_data(*cls._get_circle_points(center, r))
                theta_i = t + s * frame
                lines[i].set_data(*cls._get_arrow_points(center, r, theta_i))
                center += r * np.exp(1j * theta_i)
            trace_points.append(center)
            trace.set_data(
                [p.real for p in trace_points],
                [p.imag for p in trace_points],
            )
            return circles + lines + [trace]

        ani = FuncAnimation(
            fig,
            update,
            frames=frame_gen(),
            init_func=init_func,
            interval=interval,
            blit=True,
            repeat=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        if cls._is_jupyter():
            from IPython.display import Image
            from IPython.display import display as _display

            ani.save(gif_name, writer=PillowWriter(fps=30))
            _display(Image(filename=gif_name))
        else:
            plt.show()

    @classmethod
    def animate_discomposition(
        cls,
        n_polygon_points: int = 30,
        n_sample_points: int = 80,
        n_fft_points: int = 100,
        interval: float = 200,
        wave_n_points: int = 1000,
        gif_name: str = 'discomposite_animation.gif',
    ):
        """Animate per-mode x/y wave decomposition from FFT components.

        Args:
            n_polygon_points (int): Number of vertices in the random polygon.
            n_sample_points (int): Resampled points for FFT.
            n_fft_points (int): Number of FFT modes to use.
            interval (float): Delay between frames (ms).
            wave_n_points (int): Number of points to plot per wave.
            gif_name (str): GIF filename when saving in Jupyter.
        """
        polygon_points = cls._generate_polygon_points(n_polygon_points)
        sample_points = cls._resample_polygon_points(polygon_points, n_sample_points=n_sample_points)
        fft_points, radius, theta, speed = cls._fft_sample_points(sample_points, n_fft_points=n_fft_points)

        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(nrows=2, ncols=3)
        ax_polygon = fig.add_subplot(gs[:, 0])
        ax_x = fig.add_subplot(gs[0, 1])
        ax_y = fig.add_subplot(gs[0, 2])
        ax_xs = fig.add_subplot(gs[1, 1])
        ax_ys = fig.add_subplot(gs[1, 2])

        ax_polygon.plot(polygon_points.real, polygon_points.imag, marker='o', color='tab:blue', lw=1, markersize=2)
        ax_polygon.scatter(sample_points.real, sample_points.imag, marker='o', color='tab:orange', s=5)
        ax_x.plot(sample_points.real, marker='o', color='tab:green', lw=1, markersize=2)
        ax_x.set_title('x sample points')
        ax_xs.set_xlim(0 - wave_n_points * 0.05, wave_n_points * 1.05)
        ax_xs.set_ylim(ax_x.get_ylim())
        ax_y.plot(sample_points.imag, marker='o', color='tab:purple', lw=1, markersize=2)
        ax_y.set_title('y sample points')
        ax_ys.set_xlim(0 - wave_n_points * 0.05, wave_n_points * 1.05)
        ax_ys.set_ylim(ax_y.get_ylim())

        line_xs = []
        line_ys = []
        for _ in range(len(theta)):
            (line_x,) = ax_xs.plot([], [], lw=1)
            line_xs.append(line_x)
            (line_y,) = ax_ys.plot([], [], lw=1)
            line_ys.append(line_y)

        (line_x_sum,) = ax_xs.plot([], [], color='tab:green', lw=1)
        (line_y_sum,) = ax_ys.plot([], [], color='tab:purple', lw=1)
        xs = []
        ys = []
        for t, s, r in zip(theta, speed, radius, strict=True):
            phases = np.linspace(t, t + s * len(fft_points), wave_n_points)
            xs.append(np.cos(phases) * r)
            ys.append(np.sin(phases) * r)

        def init_func():
            for line in line_xs + line_ys:
                line.set_data([], [])
            line_x_sum.set_data([], [])
            line_y_sum.set_data([], [])
            return line_xs + line_ys + [line_x_sum, line_y_sum]

        def update(frame):
            line_xs[frame].set_data(range(wave_n_points), xs[frame])
            line_ys[frame].set_data(range(wave_n_points), ys[frame])
            line_x_sum.set_data(range(wave_n_points), np.sum(np.asarray(xs[: frame + 1]), axis=0))
            line_y_sum.set_data(range(wave_n_points), np.sum(np.asarray(ys[: frame + 1]), axis=0))
            return line_xs + line_ys + [line_x_sum, line_y_sum]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(theta),
            init_func=init_func,
            interval=interval,
            blit=True,
            repeat=False,
        )
        plt.tight_layout()
        if cls._is_jupyter():
            from IPython.display import Image
            from IPython.display import display as _display

            ani.save(gif_name, writer=PillowWriter(fps=10))
            _display(Image(filename=gif_name))
        else:
            plt.show()


if __name__ == '__main__':
    epicycle = Epicycle()
    epicycle.animate_polygon_points(
        n_polygon_points=30,
        n_sample_points=180,
        n_fft_points=80,
        n_frames=10000,
        interval=10,
    )
    epicycle.animate_discomposition(
        n_polygon_points=30,
        n_sample_points=180,
        n_fft_points=80,
        interval=200,
        wave_n_points=1000,
    )
