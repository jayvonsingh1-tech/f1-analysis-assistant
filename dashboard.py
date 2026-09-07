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
        self.animation = None

        self.grid = GridSpec(3, 12, figure=self.fig,
                             hspace=0.45, wspace=0.7,
                             left=0.03, right=0.96,
                             top=0.88, bottom=0.07)

        self.main_info = self.fig.add_subplot(self.grid[0:2, 0:3])
        self.main = self.fig.add_subplot(self.grid[0:2, 3:7])
        self.strip = self.fig.add_subplot(self.grid[2, 0:7])
        self.side = [
            self.fig.add_subplot(self.grid[0, 7:12]),
            self.fig.add_subplot(self.grid[1, 7:12]),
            self.fig.add_subplot(self.grid[2, 7:12]),
        ]

        self.clear()
        style.title(self.fig, self.title)

    def clear(self):
        """Stop any animation, then blank every panel."""
        self.stop_animation()
        for ax in [self.main, self.main_info, self.strip] + self.side:
            ax.clear()
            ax.axis('off')

    def stop_animation(self):
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

    def clear_panel(self, ax):
        if ax is self.main:
            self.stop_animation()
        ax.clear()
        ax.axis('off')

    def set_title(self, title):
        self.title = title
        style.title(self.fig, title)

    def show(self):
        telemetry.display()

# --- Shared board for the agent -------------------------------------------

_board = None

def resolve_panel(requested, default):
    """Decide which panel to use.

    If the caller named one, use it. If not, put the first visual in main
    and later ones in their default slot.
    """
    if requested:
        return requested

    board = get_board()
    occupied = any(
        len(ax.lines) or len(ax.patches) or len(ax.collections) or len(ax.images)
        for ax in [board.main] + board.side + [board.strip]
    )
    return default if occupied else 'main'

def get_board(title=None):
    """Return the shared dashboard, creating or recreating it if needed."""
    global _board
    if _board is None or not plt.fignum_exists(_board.fig.number):
        _board = Dashboard(title or "F1 Analysis")
    elif title and title != _board.title:
        _board.set_title(title)
    return _board

def panel_for(name):
    """Resolve a panel name to its axes."""
    board = get_board()
    slots = {
        'main': board.main,
        'side1': board.side[0],
        'side2': board.side[1],
        'side3': board.side[2],
        'strip': board.strip,
    }
    if name not in slots:
        raise ValueError(f"Unknown panel '{name}'. "
                         f"Use main, side1, side2, side3 or strip.")
    return slots[name]

def reset(title=None):
    """Clear every panel, optionally retitling."""
    board = get_board()
    board.clear()
    if title:
        board.set_title(title)
    return board

if __name__ == '__main__':
    board = Dashboard("Monza 2024")

    telemetry.animate_head_to_head(2024, 'Monza', 'NOR', 'PIA',
                                   ax=board.main, info_ax=board.main_info)
    telemetry.plot_gap_to_leader(2024, 'Monza', ax=board.strip)
    telemetry.plot_strategy(2024, 'Monza', ax=board.side[0])
    telemetry.plot_positions(2024, 'Monza', ax=board.side[1])
    telemetry.corner_analysis(2024, 'Monza', 'NOR', 'PIA',
                              ax=board.side[2])

    board.show()
    plt.show()