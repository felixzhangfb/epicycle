import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
    def get_circle_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        n_points: int = 100,
    ) -> tuple[NDArray, NDArray]:
        """Generate coordinate points on a circle.

        This method calculates and returns coordinate points on the circumference
        of a circle with specified center, radius, and number of points.

        Args:
            center: Center coordinates of the circle, default is origin (0+0j)
            radius: Radius of the circle, default is 1.0
            n_points: Number of points on the circle, default is 100.
                     More points result in a smoother circle

        Returns:
            tuple[NDArray, NDArray]: A tuple containing x and y coordinates
                - First array contains x coordinates
                - Second array contains y coordinates

        Example:
            >>> x, y = Epicycle.get_circle_points(center=1+1j, radius=2.0, n_points=50)
            >>> len(x)
            50
            >>> len(y)
            50
        """
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
    ) -> tuple[NDArray, NDArray]:
        """Generate coordinate points for a circle arrow (radius line).

        This method calculates and returns coordinate points for a line segment
        from the circle center to a specified angle position on the circumference.

        Args:
            center: Center coordinates of the circle, default is origin (0+0j)
            radius: Radius of the circle, default is 1.0
            theta: Angle in radians, default is 0.0, representing angle from positive x-axis

        Returns:
            tuple[NDArray, NDArray]: A tuple containing start and end points of the arrow line
                - First array contains two x coordinate values: [start_x, end_x]
                - Second array contains two y coordinate values: [start_y, end_y]

        Example:
            >>> x, y = Epicycle.get_arrow_points(center=0+0j, radius=2.0, theta=np.pi/2)
            >>> x
            array([0., 0.])
            >>> y
            array([0., 2.])
        """
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
        """Draw a static plot of a single circle.

        This method creates a matplotlib figure displaying a circle with specified
        center, radius, and angle, including the circle outline and a radius line
        (arrow) from center to the specified angle position.

        Args:
            center: Center coordinates of the circle, default is origin (0+0j)
            radius: Radius of the circle, default is 1.0
            theta: Angle in radians, default is 0.0, specifies the radius line direction
            figsize: Figure size, default is (6, 6), in inches

        Returns:
            None: This method displays the figure directly and returns no value

        Example:
            >>> epicycle = Epicycle()
            >>> epicycle.draw_circle(center=1+1j, radius=2.0, theta=np.pi/4)
            # Displays a circle with center at (1,1), radius 2, angle 45 degrees
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.plot(*cls.get_circle_points(center, radius), color='#cccccc80', lw=5)
        ax.plot(*cls.get_arrow_points(center, radius, theta), color='tab:blue', lw=2)
        fig.show()

    @classmethod
    def draw_circles(
        cls,
        radius: list[float],
        theta: list[float],
        orig: complex = 0 + 0j,
    ):
        """Draw a static plot of multiple connected circles (epicycle structure).

        This method creates a matplotlib figure displaying an epicycle structure
        composed of multiple circles. Each circle's center lies on the circumference
        of the previous circle, forming a chain-like structure.

        Args:
            radius: List of radii for each circle, should be same length as theta list
            theta: List of initial angles for each circle (in radians),
                  should be same length as radius list
            orig: Starting origin coordinates, default is (0+0j).
                 Center position of the first circle

        Returns:
            None: This method displays the figure directly and returns no value

        Example:
            >>> epicycle = Epicycle()
            >>> epicycle.draw_circles(
            ...     radius=[3.0, 2.0, 1.0],
            ...     theta=[0.0, np.pi/2, np.pi],
            ...     orig=0+0j
            ... )
            # Draws three connected circles
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.set_axis_off()
        center = orig
        for r, t in zip(radius, theta, strict=True):
            ax.plot(*cls.get_circle_points(center, r), color='#cccccc80', lw=5)
            ax.plot(*cls.get_arrow_points(center, r, t), color='tab:blue', lw=2)
            center += r * np.exp(1j * t)
        fig.show()

    @classmethod
    def animate_circle(
        cls,
        center: complex,
        radius: float,
        speed: float,
        frames: int = 360,
    ):
        """Create an animation of a single circle rotating.

        This method creates an animation showing a circle rotating around its center,
        while recording and displaying the motion trajectory of a point on the
        circumference (red trajectory line).

        Args:
            center: Center coordinates of the circle, cannot be default value
            radius: Radius of the circle, cannot be default value
            speed: Animation speed control, smaller values make animation faster
            frames: Number of animation frames, default is 360 (one full rotation)

        Returns:
            None: This method displays the animation directly and returns no value

        Animation Features:
            - Circle outline displayed in gray
            - Radius line displayed in steel blue
            - Trajectory line displayed in red
            - Animation rotates in clockwise direction

        Example:
            >>> epicycle = Epicycle()
            >>> epicycle.animate_circle(center=0+0j, radius=2.0, speed=0.1, frames=720)
            # Creates a circle with radius 2 rotating two full cycles
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
        cls,
        radius: list[float],
        theta: list[float],
        speed: list[float],
        orig: complex = 0 + 0j,
        frames: int = 360,
        interval: float = 0.1,
    ):
        """Create an animation of multiple connected circles (epicycle).

        This method creates a complex animation displaying an epicycle structure
        composed of multiple circles. Each circle rotates around its center, and
        each circle's center lies on the circumference of the previous circle,
        forming a chain-like motion structure. It also records and displays the
        motion trajectory of the final endpoint.

        Args:
            radius: List of radii for each circle, should be same length as theta and speed lists
            theta: List of initial angles for each circle (in radians),
                  should be same length as radius and speed lists
            speed: List of rotation speeds for each circle (radians/frame),
                  should be same length as radius and theta lists
            orig: Starting origin coordinates, default is (0+0j).
                 Center position of the first circle
            frames: Total number of animation frames, default is 360
            interval: Time interval between frames in seconds, default is 0.1

        Returns:
            None: This method displays the animation directly and returns no value

        Animation Features:
            - Circle outlines displayed as gray thin lines
            - Radius lines displayed in steel blue
            - Final endpoint trajectory displayed in red
            - Animation automatically stops when trajectory approaches starting point (closed loop detection)
            - All circles rotate simultaneously at their respective speeds

        Performance Notes:
            - This method prints debug information showing input parameters
            - Animation automatically stops when closed loop is detected to avoid infinite loops

        Example:
            >>> epicycle = Epicycle()
            >>> epicycle.animate_circles(
            ...     radius=[3.0, 2.0, 1.0],
            ...     theta=[0.0, np.pi/4, np.pi/2],
            ...     speed=[0.01, 0.02, 0.03],
            ...     orig=0+0j,
            ...     frames=1000
            ... )
            # Creates an animation of three connected circles with different speeds and radiuses
        """
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

        def update(frame):
            center = orig
            for i, (r, t, s) in enumerate(zip(radius, theta, speed, strict=True)):
                circles[i].set_data(*cls.get_circle_points(center, r))
                # compute angle for this frame without mutating the original `theta` list
                theta_i = t + s * frame
                lines[i].set_data(*cls.get_arrow_points(center, r, theta_i))
                center += r * np.exp(1j * theta_i)
            if len(trace_points) > 2 and np.abs(center - trace_points[0]) < 1e-10:
                ani.event_source.stop()
            trace_points.append(center)
            trace.set_data(
                [p.real for p in trace_points],
                [p.imag for p in trace_points],
            )
            return circles + lines + [trace]

        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init_func,
            interval=interval,
            blit=True,
            repeat=False,
        )
        plt.tight_layout()
        plt.show()
        # plt.close(fig)
        # with open('animation.html', 'w') as f:
        #     f.write(ani.to_jshtml())

    @staticmethod
    def generate_polygon_points(n_polygon_points: int = 20) -> NDArray[np.complex128]:
        z = np.random.randint(-10, 10, n_polygon_points) + 1j * np.random.randint(-10, 10, n_polygon_points)
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
    def resample_polygon_points(
        points: NDArray[np.complex128],
        n_sample_points: int = 50,
    ) -> NDArray[np.complex128]:
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
    def fft_sample_points(
        sample_points: NDArray[np.complex128],
        n_fft_points: int = 100,
    ) -> tuple[
        NDArray[np.complex128],
        list[float],
        list[float],
        list[float],
    ]:
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
    ):
        polygon_points = cls.generate_polygon_points(n_polygon_points)
        sample_points = cls.resample_polygon_points(polygon_points, n_sample_points=n_sample_points)
        fft_points, radius, theta, speed = cls.fft_sample_points(sample_points, n_fft_points=n_fft_points)

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

        ax_polygon.plot(polygon_points.real, polygon_points.imag, marker='o', color='tab:blue', lw=1, markersize=2)
        xlim = ax_polygon.get_xlim()
        ylim = ax_polygon.get_ylim()

        ax_sample.plot(sample_points.real, sample_points.imag, marker='o', color='tab:orange', lw=1, markersize=2)
        ax_fft.plot(fft_points.real, fft_points.imag, marker='o', color='tab:green', lw=1, markersize=2)

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
                if len(trace_points) > 2 and np.abs(trace_points[-1] - trace_points[0]) < 1e-10:
                    break
                yield i

        def update(frame):
            center = 0 + 0j
            for i, (r, t, s) in enumerate(zip(radius, theta, speed, strict=True)):
                circles[i].set_data(*cls.get_circle_points(center, r))
                theta_i = t + s * frame * 0.5
                lines[i].set_data(*cls.get_arrow_points(center, r, theta_i))
                center += r * np.exp(1j * theta_i)
            # if len(trace_points) > 2 and np.abs(trace_points[-1] - trace_points[0]) < 1e-10:
            #     ani.event_source.stop()
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
        # plt.show()
        ani.save('animate_polygon_points.mp4', writer='ffmpeg', dpi=150, fps=30)
        plt.close(fig)


if __name__ == '__main__':
    epicycle = Epicycle()
    epicycle.animate_polygon_points(
        n_polygon_points=30,
        n_sample_points=180,
        n_fft_points=80,
        n_frames=10000,
        interval=10,
    )
