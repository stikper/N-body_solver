from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .base import Visualizer


class Animator2D(Visualizer):
    def __init__(
            self,
            positions_history: np.ndarray,
            masses: np.ndarray,
            dt: float,
            particle_names: Optional[List[str]] = None,
            particle_colors: Optional[List[str]] = None,
            particle_sizes: Optional[List[float]] = None,
            fig_size: Tuple[float, float] = (10, 10),
            x_lim: Optional[Tuple[float, float]] = None,
            y_lim: Optional[Tuple[float, float]] = None,
            title: str = '🌌 N-Body Simulation',
            show_trails: bool = True,
            trail_length: Optional[int] = None,
            dark_theme: bool = True,
    ):
        self.positions_history = positions_history
        self.masses = masses
        self.dt = dt
        self.n_steps, self.n_particles, _ = positions_history.shape

        # Particle names
        if particle_names is None:
            self.particle_names = [f'Particle {i}' for i in range(self.n_particles)]
        else:
            self.particle_names = particle_names

        # Particle colors
        if particle_colors is None:
            default_colors = ['yellow', 'dodgerblue', 'red', 'green',
                              'orange', 'purple', 'pink', 'cyan']
            self.particle_colors = [
                default_colors[i % len(default_colors)]
                for i in range(self.n_particles)
            ]
        else:
            self.particle_colors = particle_colors

        # Particle size (proportional to mass)
        if particle_sizes is None:
            mass_ratios = self.masses / np.max(self.masses)
            self.particle_sizes = 5 + 15 * mass_ratios.flatten()  # от 5 до 20
        else:
            self.particle_sizes = particle_sizes

        self.fig_size = fig_size
        self.title = title
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.dark_theme = dark_theme

        # Axis limits
        if x_lim is None:
            x_positions = positions_history[:, :, 0]
            margin = 0.2 * (x_positions.max() - x_positions.min())
            self.x_lim = (x_positions.min() - margin, x_positions.max() + margin)
        else:
            self.x_lim = x_lim

        if y_lim is None:
            y_positions = positions_history[:, :, 1]
            margin = 0.2 * (y_positions.max() - y_positions.min())
            self.y_lim = (y_positions.min() - margin, y_positions.max() + margin)
        else:
            self.y_lim = y_lim

        # Animation attributes
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.anim: Optional[FuncAnimation] = None
        self._particles: List = []
        self._trails: List = []
        self._time_text = None

    def _setup_figure(self):
        """Настройка фигуры и осей (внутренний метод)"""
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)

        # Axis settings
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_aspect('equal')

        # Dark theme
        if self.dark_theme:
            self.ax.set_facecolor('#000814')
            self.fig.patch.set_facecolor('#001d3d')
            self.ax.grid(True, alpha=0.2, color='cyan')
            self.ax.set_xlabel('X [м]', color='white', fontsize=12)
            self.ax.set_ylabel('Y [м]', color='white', fontsize=12)
            self.ax.set_title(self.title, color='yellow', fontsize=16, weight='bold')
            self.ax.tick_params(colors='white')
            legend_kwargs = {
                'facecolor': 'black',
                'edgecolor': 'cyan',
                'labelcolor': 'white'
            }
        else:
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X [м]', fontsize=12)
            self.ax.set_ylabel('Y [м]', fontsize=12)
            self.ax.set_title(self.title, fontsize=16, weight='bold')
            legend_kwargs = {}

        # Create particle markers
        self._particles = []
        for i in range(self.n_particles):
            particle, = self.ax.plot(
                [], [], 'o',
                color=self.particle_colors[i],
                markersize=self.particle_sizes[i],
                label=self.particle_names[i],
                zorder=10 + i
            )
            self._particles.append(particle)

        # Create trajectory
        if self.show_trails:
            self._trails = []
            for i in range(self.n_particles):
                trail, = self.ax.plot(
                    [], [], '-',
                    color=self.particle_colors[i],
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=i
                )
                self._trails.append(trail)

        text_color = 'white' if self.dark_theme else 'black'
        bbox_color = 'black' if self.dark_theme else 'white'
        self._time_text = self.ax.text(
            0.02, 0.95, '',
            transform=self.ax.transAxes,
            color=text_color,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7)
        )

        self.ax.legend(loc='upper right', fontsize=10, **legend_kwargs)

    def _update_frame(self, frame: int):
        artists = []

        # Update particle position
        for i, particle in enumerate(self._particles):
            pos = self.positions_history[frame, i, :2]
            particle.set_data([pos[0]], [pos[1]])
            artists.append(particle)

        # Update trajectory
        if self.show_trails:
            for i, trail in enumerate(self._trails):
                if self.trail_length is None:
                    # All trajectory
                    trail_data = self.positions_history[:frame + 1, i, :2]
                else:
                    # Trail limit
                    start = max(0, frame - self.trail_length)
                    trail_data = self.positions_history[start:frame + 1, i, :2]

                trail.set_data(trail_data[:, 0], trail_data[:, 1])
                artists.append(trail)

        # Text update
        time_days = frame * self.dt / 86400.0
        self._time_text.set_text(
            f'День: {time_days:.1f}\n'
            f'Шаг: {frame}/{self.n_steps}'
        )
        artists.append(self._time_text)

        return artists

    def visualize(
            self,
            interval: float = 1,
            repeat: bool = True,
            show: bool = True
    ):
        self._setup_figure()

        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.n_steps,
            interval=interval,
            blit=True,
            repeat=repeat
        )

        if show:
            plt.show()

    def save(
            self,
            filename: str,
            fps: int = 30,
            dpi: int = 150,
            writer: str = 'ffmpeg',
            **kwargs
    ):
        if self.anim is None:
            print("️⚠️ Call visualize() first, motherfucker!")
            return

        print(f" Saving animation to '{filename}'...")
        self.anim.save(filename, writer=writer, fps=fps, dpi=dpi, **kwargs)
        print(f"✅ Saved to '{filename}'!")