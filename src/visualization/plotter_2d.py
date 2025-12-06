from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .base import Visualizer


class Plotter2D(Visualizer):

    def __init__(
            self,
            positions_history: np.ndarray,
            masses: np.ndarray,
            dt: float,
            particle_names: Optional[List[str]] = None,
            particle_colors: Optional[List[str]] = None,
            figsize: Tuple[float, float] = (10, 10),
            title: str = '🌌 Particle Trajectories',
            dark_theme: bool = True,
    ):
        self.positions_history = positions_history
        self.masses = masses
        self.dt = dt
        self.n_steps, self.n_particles, _ = positions_history.shape

        if particle_names is None:
            self.particle_names = [f'Particle {i}' for i in range(self.n_particles)]
        else:
            self.particle_names = particle_names

        if particle_colors is None:
            default_colors = ['yellow', 'dodgerblue', 'red', 'green',
                              'orange', 'purple', 'pink', 'cyan']
            self.particle_colors = [
                default_colors[i % len(default_colors)]
                for i in range(self.n_particles)
            ]
        else:
            self.particle_colors = particle_colors

        self.figsize = figsize
        self.title = title
        self.dark_theme = dark_theme

        self.fig = None
        self.ax = None

    def visualize(
            self,
            show_start: bool = True,
            show_end: bool = True,
            show: bool = True
    ):
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Dark theme
        if self.dark_theme:
            self.ax.set_facecolor('#000814')
            self.fig.patch.set_facecolor('#001d3d')
            self.ax.grid(True, alpha=0.2, color='cyan')
            self.ax.set_xlabel('X [m]', color='white', fontsize=12)
            self.ax.set_ylabel('Y [m]', color='white', fontsize=12)
            self.ax.set_title(self.title, color='yellow', fontsize=16, weight='bold')
            self.ax.tick_params(colors='white')
            legend_kwargs = {'facecolor': 'black', 'edgecolor': 'cyan', 'labelcolor': 'white'}
        else:
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X [m]', fontsize=12)
            self.ax.set_ylabel('Y [m]', fontsize=12)
            self.ax.set_title(self.title, fontsize=16, weight='bold')
            legend_kwargs = {}

        # Plot trajectories
        for i in range(self.n_particles):
            trajectory = self.positions_history[:, i, :2]

            # Plot trajectory line
            self.ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                '-',
                color=self.particle_colors[i],
                linewidth=1.5,
                alpha=0.7,
                label=self.particle_names[i]
            )

            # Mark start position
            if show_start:
                self.ax.plot(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    'o',
                    color=self.particle_colors[i],
                    markersize=8,
                    markeredgecolor='white' if self.dark_theme else 'black',
                    markeredgewidth=1.5,
                    zorder=10
                )

            # Mark end position
            if show_end:
                self.ax.plot(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    's',
                    color=self.particle_colors[i],
                    markersize=8,
                    markeredgecolor='white' if self.dark_theme else 'black',
                    markeredgewidth=1.5,
                    zorder=10
                )

        self.ax.set_aspect('equal')
        self.ax.legend(loc='best', fontsize=10, **legend_kwargs)

        if show:
            plt.show()

    def save(self, filename: str, dpi: int = 300, **kwargs):
        if self.fig is None:
            print("⚠️  Call visualize() first, motherfucker!")
            return

        print(f"💾 Saving plot to '{filename}'...")
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"✅ Saved to '{filename}'!")