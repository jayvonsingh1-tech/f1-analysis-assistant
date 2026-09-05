"""Multi-panel dashboard for F1 analysis."""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style
import telemetry

style.apply()

class Dashboard:
    def __init__(self, title="F1 Analysis"):
        self.fig = plt.figure(figsize=(19, 10))
        self.title = title

        # 12 columns gives fine control over the split.
        # Main takes 8 of 12 across and 2 of 3 rows down.
        # The strip sits beneath it; side panels fill the right column.
        self.grid = GridSpec(3, 12, figure=self.fig,
                             hspace=0.45, wspace=0.6,
                             left=0.03, right=0.97,
                             top=0.88, bottom=0.06)

        self.main = self.fig.add_subplot(self.grid[0:2, 0:8])
        self.strip = self.fig.add_subplot(self.grid[2, 0:8])
        self.side = [
            self.fig.add_subplot(self.grid[0, 8:12]),
            self.fig.add_subplot(self.grid[1, 8:12]),
            self.fig.add_subplot(self.grid[2, 8:12]),
        ]

        self.clear()
        style.title(self.fig, self.title)

    def clear(self):
        """Blank every panel."""
        for ax in [self.main, self.strip] + self.side:
            ax.clear()
            ax.axis('off')

    def clear_panel(self, ax):
        """Blank one panel."""
        ax.clear()
        ax.axis('off')

    def set_title(self, title):
        self.title = title
        style.title(self.fig, title)

    def show(self):
        telemetry.display()

if __name__ == '__main__':
    board = Dashboard("Monza 2024")

    telemetry.animate_head_to_head(2024, 'Monza', 'NOR', 'PIA',
                                   ax=board.main)
    telemetry.plot_gap_to_leader(2024, 'Monza', ax=board.strip)

    telemetry.plot_strategy(2024, 'Monza', ax=board.side[0])
    telemetry.plot_positions(2024, 'Monza', ax=board.side[1])
    telemetry.corner_analysis(2024, 'Monza', 'NOR', 'PIA',
                              ax=board.side[2])

    board.show()
    plt.show()
    