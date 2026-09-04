"""Shared visual style for all F1 project charts."""

import matplotlib.pyplot as plt

# Palette
BACKGROUND = '#0E0E14'
PANEL = '#16161F'
TEXT = '#E8E8EC'
MUTED = '#8A8A99'
GRID = '#2A2A38'
TRACK = '#3A3A48'
ACCENT = '#E10600'

TEAM_COLOURS = {
    'McLaren': '#FF8000',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'Red Bull Racing': '#3671C6',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'Haas F1 Team': '#B6BABD',
    'Racing Bulls': '#6692FF',
    'Audi': '#00E701',
    'Kick Sauber': '#52E252',
    'Cadillac': '#C4A000',
}

COMPOUND_COLOURS = {
    'SOFT': '#DA291C',
    'MEDIUM': '#FFD12E',
    'HARD': '#C4C4C8',
    'INTERMEDIATE': '#43B02A',
    'WET': '#0067AD',
    'UNKNOWN': '#5A5A66',
}

# Two-driver comparison colours, used when team colours would clash
DRIVER_A = '#FF8000'
DRIVER_B = '#27F4D2'

def apply():
    """Apply the project style to all subsequent matplotlib figures."""
    plt.rcParams.update({
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'savefig.facecolor': BACKGROUND,

        'text.color': TEXT,
        'axes.labelcolor': MUTED,
        'xtick.color': MUTED,
        'ytick.color': MUTED,

        'axes.edgecolor': GRID,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,

        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.6,
        'axes.grid': True,

        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'figure.titlesize': 16,

        'legend.frameon': False,
        'legend.labelcolor': TEXT,

        'lines.linewidth': 2,
        'lines.antialiased': True,
    })

def title(fig, main, subtitle=None):
    """Consistent title block: bold main line, muted subtitle beneath."""
    fig.suptitle(main, color=TEXT, fontsize=16, fontweight='bold',
                 x=0.02, ha='left', y=0.97)
    if subtitle:
        fig.text(0.02, 0.925, subtitle, color=MUTED, fontsize=11, ha='left')
        