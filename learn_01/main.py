from random import randint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Epicycle:
    """一个用于创建和动画演示 epicycle（圆内旋轮线）的类。

    Epicycle（圆内旋轮线）是由多个圆组成的数学曲线，其中每个圆的中心
    位于前一个圆的圆周上。该类提供了创建、绘制和动画化这些几何图形的功能。

    主要功能包括：
    - 生成圆的坐标点
    - 绘制静态圆形和圆形链
    - 创建单个圆和多个圆的动画效果
    - 可视化 epicycle 的运动轨迹

    示例：
        >>> epicycle = Epicycle()
        >>> epicycle.draw_circle(radius=2.0, theta=0.5)
        >>> epicycle.animate_circles([1, 2, 3], [0, 1, 2], [1, 0.5, 0.25])
    """

    @staticmethod
    def get_circle_points(
        center: complex = 0 + 0j,
        radius: float = 1.0,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成圆形上的一系列坐标点。

        该方法计算并返回指定圆心、半径和点数的圆形轨迹上的坐标点。

        Args:
            center: 圆心坐标，默认为原点 (0+0j)
            radius: 圆的半径，默认为 1.0
            n_points: 圆形上点的数量，默认为 100。点数越多，圆形越平滑

        Returns:
            tuple[np.ndarray, np.ndarray]: 包含 x 坐标和 y 坐标的元组
                - 第一个数组包含 x 坐标
                - 第二个数组包含 y 坐标

        示例：
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成圆形上箭头（半径线）的坐标点。

        该方法计算并返回从圆心到圆周上指定角度位置的箭头线段坐标点。

        Args:
            center: 圆心坐标，默认为原点 (0+0j)
            radius: 圆的半径，默认为 1.0
            theta: 角度（弧度），默认为 0.0，表示从正x轴开始的角度

        Returns:
            tuple[np.ndarray, np.ndarray]: 包含箭头线段起点和终点的坐标
                - 第一个数组包含两个x坐标值：[起点x, 终点x]
                - 第二个数组包含两个y坐标值：[起点y, 终点y]

        示例：
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
        """绘制单个圆形的静态图形。

        该方法创建一个 matplotlib 图形，显示指定圆心、半径和角度的圆形，
        包括圆形轮廓线和从圆心到圆周上指定角度位置的半径线（箭头）。

        Args:
            center: 圆心坐标，默认为原点 (0+0j)
            radius: 圆的半径，默认为 1.0
            theta: 角度（弧度），默认为 0.0，指定半径线的朝向
            figsize: 图形大小，默认为 (6, 6)，单位为英寸

        Returns:
            None: 该方法直接显示图形，不返回值

        示例：
            >>> epicycle = Epicycle()
            >>> epicycle.draw_circle(center=1+1j, radius=2.0, theta=np.pi/4)
            # 显示一个圆心在(1,1)，半径为2，角度为45度的圆形
        """
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
        """绘制多个连接的圆形（epicycle 结构）的静态图形。

        该方法创建一个 matplotlib 图形，显示由多个圆形组成的 epicycle 结构。
        每个圆的中心都位于前一个圆的圆周上，形成链状结构。

        Args:
            radius: 每个圆的半径列表，长度应该与 theta 列表相同
            theta: 每个圆的初始角度列表（弧度），长度应该与 radius 列表相同
            orig: 起始原点坐标，默认为 (0+0j)。第一个圆的圆心位置

        Returns:
            None: 该方法直接显示图形，不返回值

        示例：
            >>> epicycle = Epicycle()
            >>> epicycle.draw_circles(
            ...     radius=[3.0, 2.0, 1.0],
            ...     theta=[0.0, np.pi/2, np.pi],
            ...     orig=0+0j
            ... )
            # 绘制三个连接的圆形
        """
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
    def animate_circle(
        cls,
        center: complex,
        radius: float,
        speed: float,
        frames: int = 360,
    ):
        """创建单个圆形旋转的动画。

        该方法创建一个动画，显示圆形围绕其中心旋转，同时记录并显示
        圆周上一点的运动轨迹（红色轨迹线）。

        Args:
            center: 圆心坐标，不能为默认值
            radius: 圆的半径，不能为默认值
            speed: 动画速度控制，数值越小动画越快
            frames: 动画帧数，默认为 360帧（一圈）

        Returns:
            None: 该方法直接显示动画，不返回值

        动画特性：
            - 圆形轮廓显示为灰色
            - 半径线显示为钢蓝色
            - 轨迹线显示为红色
            - 动画按顺时针方向旋转

        示例：
            >>> epicycle = Epicycle()
            >>> epicycle.animate_circle(center=0+0j, radius=2.0, speed=0.1, frames=720)
            # 创建半径为2的圆的两圈旋转动画
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        # ax.set_axis_off()
        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
        (circle,) = ax.plot([], [], color='gray', lw=10, alpha=0.5)
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
        cls,
        radius: list[float],
        theta: list[float],
        speed: list[float],
        orig: complex = 0 + 0j,
        frames: int = 360,
    ):
        """创建多个连接圆形（epicycle）的旋转动画。

        该方法创建一个复杂的动画，显示多个圆形组成的 epicycle 结构。
        每个圆都围绕其中心旋转，并且每个圆的中心都位于前一个圆的圆周上，
        形成链状运动结构。同时记录并显示最终端点的运动轨迹。

        Args:
            radius: 每个圆的半径列表，长度应该与 theta 和 speed 列表相同
            theta: 每个圆的初始角度列表（弧度），长度应该与 radius 和 speed 列表相同
            speed: 每个圆的旋转速度列表（弧度/帧），长度应该与 radius 和 theta 列表相同
            orig: 起始原点坐标，默认为 (0+0j)。第一个圆的圆心位置
            frames: 动画总帧数，默认为 360帧

        Returns:
            None: 该方法直接显示动画，不返回值

        动画特性：
            - 圆形轮廓显示为灰色细线
            - 半径线显示为钢蓝色
            - 最终端点轨迹显示为红色
            - 当轨迹接近起始点时动画自动停止（检测闭环）
            - 所有圆同时按各自的速度旋转

        性能说明：
            - 该方法会打印调试信息显示输入参数
            - 动画会在检测到闭环时自动停止以避免无限循环

        示例：
            >>> epicycle = Epicycle()
            >>> epicycle.animate_circles(
            ...     radius=[3.0, 2.0, 1.0],
            ...     theta=[0.0, np.pi/4, np.pi/2],
            ...     speed=[0.01, 0.02, 0.03],
            ...     orig=0+0j,
            ...     frames=1000
            ... )
            # 创建三个不同速度和半径的连接圆形的动画
        """
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
        (trace,) = ax.plot([], [], color='tab:red', lw=1)

        def init_func():
            center = orig
            t0 = 0
            for r, t in zip(radius, theta, strict=True):
                (circle,) = ax.plot([], [], color='gray', ls='-', lw=1, alpha=0.5)
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
            if len(trace_points) > 2 and np.abs(center - trace_points[0]) < 1e-10:
                ani.event_source.stop()
            trace_points.append(center)
            trace.set_data([p.real for p in trace_points], [p.imag for p in trace_points])
            return circles + lines + [trace]

        ani = FuncAnimation(
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
    K = 10
    epicycle.animate_circles(
        # radius=sorted([randint(1, 10) for _ in range(K)], reverse=True),
        radius=[randint(1, 10) for _ in range(K)],
        theta=[randint(1, 10) / 10 for _ in range(K)],
        speed=[randint(-10, 10) / 10 for _ in range(K)],
        orig=0 + 0j,
        frames=10000,
    )
