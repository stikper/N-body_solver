from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from .base import Visualizer


class EnergyPlotter(Visualizer):

    def __init__(
            self,
            times: np.ndarray,
            kinetic_energies: np.ndarray,
            potential_energies: np.ndarray,
            total_energies: np.ndarray,
            figsize: Tuple[float, float] = (12, 8),
            title: str = '⚡ Energy Conservation',
            dark_theme: bool = True,
    ):
        self.times = times
        self.kinetic = kinetic_energies
        self.potential = potential_energies
        self.total = total_energies
        self.figsize = figsize
        self.title = title
        self.dark_theme = dark_theme

        self.fig = None
        self.axes = None

    def visualize(
            self,
            show_relative_error: bool = True,
            show: bool = True
    ):
        # Create subplots
        if show_relative_error:
            self.fig, self.axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
            ax1, ax2 = self.axes
        else:
            self.fig, ax1 = plt.subplots(figsize=self.figsize)
            self.axes = [ax1]

        # Convert time to days
        times_days = self.times / 86400.0

        # Dark theme setup
        if self.dark_theme:
            self.fig.patch.set_facecolor('#001d3d')
            for ax in self.axes:
                ax.set_facecolor('#000814')
                ax.grid(True, alpha=0.2, color='cyan')
                ax.tick_params(colors='white')
            text_color = 'white'
            legend_kwargs = {'facecolor': 'black', 'edgecolor': 'cyan', 'labelcolor': 'white'}
        else:
            for ax in self.axes:
                ax.grid(True, alpha=0.3)
            text_color = 'black'
            legend_kwargs = {}

        # Plot energies
        ax1.plot(times_days, self.kinetic, 'r-', label='Kinetic', linewidth=1.5)
        ax1.plot(times_days, self.potential, 'b-', label='Potential', linewidth=1.5)
        ax1.plot(times_days, self.total, 'g-', label='Total', linewidth=2)

        ax1.set_ylabel('Energy [J]', color=text_color, fontsize=12)
        ax1.set_title(self.title, color='yellow' if self.dark_theme else 'black',
                      fontsize=16, weight='bold')
        ax1.legend(loc='best', fontsize=10, **legend_kwargs)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # Plot relative error
        if show_relative_error:
            E0 = self.total[0]
            relative_error = np.abs((self.total - E0) / E0) * 100  # in percent

            ax2.plot(times_days, relative_error, 'm-', linewidth=1.5)
            ax2.set_xlabel('Time [days]', color=text_color, fontsize=12)
            ax2.set_ylabel('Relative Error [%]', color=text_color, fontsize=12)
            ax2.set_yscale('log')
        else:
            ax1.set_xlabel('Time [days]', color=text_color, fontsize=12)

        plt.tight_layout()

        if show:
            plt.show()

    def save(self, filename: str, dpi: int = 300, **kwargs):
        if self.fig is None:
            print("⚠️  Call visualize() first, dickhead!")
            return

        print(f"💾 Saving energy plot to '{filename}'...")
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"✅ Saved to '{filename}'!")