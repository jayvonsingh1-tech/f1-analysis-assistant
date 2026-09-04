import logging

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

COMPOUND_COLOURS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFF200',
    'HARD': '#EBEBEB',
    'INTERMEDIATE': '#39B54A',
    'WET': '#00AEEF',
}

def get_lap_telemetry(year, race, driver, lap_number=None, session_type='R'):
    """Return (lap, telemetry) for one lap. Fastest lap if lap_number is None."""
    session = fastf1.get_session(year, race, session_type)
    session.load()

    laps = session.laps.pick_drivers(driver)

    if lap_number is None:
        lap = laps.pick_fastest()
    else:
        lap = laps[laps['LapNumber'] == lap_number].iloc[0]

    telemetry = lap.get_telemetry().add_distance()

    return lap, telemetry

def plot_lap(year, race, driver, lap_number=None):
    lap, tel = get_lap_telemetry(year, race, driver, lap_number)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(tel['Distance'], tel['Speed'])
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(tel['Distance'], tel['Throttle'])
    axes[1].set_ylabel('Throttle (%)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(tel['Distance'], tel['Brake'].astype(int))
    axes[2].set_ylabel('Brake')
    axes[2].set_xlabel('Distance (m)')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"{driver} - {race} {year} - lap {int(lap['LapNumber'])} "
                 f"({lap['LapTime']})")
    plt.tight_layout()
    plt.show()

def compare_laps(year, race, driver_a, driver_b, session_type='R'):
    lap_a, tel_a = get_lap_telemetry(year, race, driver_a, session_type=session_type)
    lap_b, tel_b = get_lap_telemetry(year, race, driver_b, session_type=session_type)

    max_distance = min(tel_a['Distance'].max(), tel_b['Distance'].max())
    grid = np.linspace(0, max_distance, 1000)

    def elapsed(tel):
        return (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds()

    time_a = np.interp(grid, tel_a['Distance'], elapsed(tel_a))
    time_b = np.interp(grid, tel_b['Distance'], elapsed(tel_b))
    delta = time_a - time_b

    speed_a = np.interp(grid, tel_a['Distance'], tel_a['Speed'])
    speed_b = np.interp(grid, tel_b['Distance'], tel_b['Speed'])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    axes[0].plot(grid, speed_a, label=driver_a)
    axes[0].plot(grid, speed_b, label=driver_b)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(grid, delta, color='black')
    axes[1].axhline(0, color='grey', linewidth=0.8)
    axes[1].fill_between(grid, delta, 0, where=delta > 0, alpha=0.3, color='red')
    axes[1].fill_between(grid, delta, 0, where=delta < 0, alpha=0.3, color='green')
    axes[1].set_ylabel(f'Delta (s)\n+ = {driver_a} slower')
    axes[1].set_xlabel('Distance (m)')
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{driver_a} vs {driver_b} - {race} {year} - fastest laps")
    plt.tight_layout()
    plt.show()

def plot_track(year, race, driver='NOR'):
    lap, tel = get_lap_telemetry(year, race, driver)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(tel['X'], tel['Y'], linewidth=8, color='grey')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{race} {year}")
    plt.show()

def plot_speed_map(year, race, driver, lap_number=None):
    """Track outline coloured by speed."""
    lap, tel = get_lap_telemetry(year, race, driver, lap_number)

    points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#15151E')
    ax.set_facecolor('#15151E')

    collection = LineCollection(segments, cmap='plasma', linewidth=6)
    collection.set_array(tel['Speed'].to_numpy()[:-1])
    ax.add_collection(collection)

    ax.set_xlim(tel['X'].min() - 500, tel['X'].max() + 500)
    ax.set_ylim(tel['Y'].min() - 500, tel['Y'].max() + 500)
    ax.set_aspect('equal')
    ax.axis('off')

    bar = fig.colorbar(collection, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label('Speed (km/h)', color='white')
    bar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(bar.ax.axes, 'yticklabels'), color='white')

    ax.set_title(f"{driver} - {race} {year} - lap "
                 f"{int(lap['LapNumber'])} ({lap['LapTime']})",
                 color='white', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_strategy(year, race):
    """Every driver's tyre stints as coloured bars."""
    session = fastf1.get_session(year, race, 'R')
    session.load()

    laps = session.laps
    order = session.results['Abbreviation'].tolist()

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#15151E')
    ax.set_facecolor('#15151E')

    for row, driver in enumerate(order):
        driver_laps = laps[laps['Driver'] == driver]
        if driver_laps.empty:
            continue

        start = 0
        for stint, stint_laps in driver_laps.groupby('Stint'):
            compound = stint_laps['Compound'].iloc[0]
            length = len(stint_laps)
            ax.barh(row, length, left=start, height=0.7,
                    color=COMPOUND_COLOURS.get(compound, '#888888'),
                    edgecolor='#15151E')
            start += length

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Lap', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour)
               for colour in COMPOUND_COLOURS.values()]
    ax.legend(handles, COMPOUND_COLOURS.keys(), loc='lower right')

    ax.set_title(f"Tyre strategy - {race} {year}", color='white', fontsize=14)
    plt.tight_layout()
    plt.show()

def animate_head_to_head(year, race, driver_a, driver_b, frames=400):
    lap_a, tel_a = get_lap_telemetry(year, race, driver_a)
    lap_b, tel_b = get_lap_telemetry(year, race, driver_b)

    duration = max(
        (tel_a['Time'] - tel_a['Time'].iloc[0]).dt.total_seconds().max(),
        (tel_b['Time'] - tel_b['Time'].iloc[0]).dt.total_seconds().max(),
    )
    timeline = np.linspace(0, duration, frames)

    def positions(tel):
        elapsed = (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds()
        x = np.interp(timeline, elapsed, tel['X'])
        y = np.interp(timeline, elapsed, tel['Y'])
        return x, y

    xa, ya = positions(tel_a)
    xb, yb = positions(tel_b)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#15151E')
    ax.set_facecolor('#15151E')

    ax.plot(tel_a['X'], tel_a['Y'], linewidth=10, color='#333333', zorder=1)
    ax.set_aspect('equal')
    ax.axis('off')

    car_a, = ax.plot([], [], 'o', markersize=14, color='#FF8000',
                     label=driver_a, zorder=3)
    car_b, = ax.plot([], [], 'o', markersize=14, color='#00D2BE',
                     label=driver_b, zorder=3)
    clock = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                    fontsize=12, va='top', family='monospace')

    ax.legend(loc='upper right')
    ax.set_title(f"{driver_a} vs {driver_b} - {race} {year}", color='white')

    def update(frame):
        car_a.set_data([xa[frame]], [ya[frame]])
        car_b.set_data([xb[frame]], [yb[frame]])
        clock.set_text(f"{timeline[frame]:5.1f}s")
        return car_a, car_b, clock

    animation = FuncAnimation(fig, update, frames=frames,
                              interval=30, blit=True, repeat=True)

    plt.show()
    return animation

if __name__ == '__main__':
    plot_speed_map(2024, 'Monza', 'NOR')
    