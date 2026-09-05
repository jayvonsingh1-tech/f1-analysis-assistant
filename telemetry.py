import logging

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import style

fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

style.apply()

def display():
    """Show a figure without blocking the caller."""
    plt.show(block=False)
    plt.pause(0.1)

def close_all():
    plt.close('all')

def load_session(year, race, session_type='R'):
    session = fastf1.get_session(year, race, session_type)
    session.load()
    return session

def get_lap_telemetry(year, race, driver, lap_number=None, session_type='R'):
    """Return (lap, telemetry, session) for one lap. Fastest lap if None."""
    session = load_session(year, race, session_type)
    laps = session.laps.pick_drivers(driver)

    if lap_number is None:
        lap = laps.pick_fastest()
    else:
        lap = laps[laps['LapNumber'] == lap_number].iloc[0]

    if lap is None:
        raise ValueError(f"{driver} has no usable lap in this session.")

    telemetry = lap.get_telemetry().add_distance()

    return lap, telemetry, session

def format_lap_time(delta):
    """Turn a pandas Timedelta into 1:21.432."""
    seconds = delta.total_seconds()
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"

def draw_start_line(ax, tel):
    ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], marker='s', markersize=9,
            color=style.TEXT, zorder=6)
    ax.annotate('LAP START', (tel['X'].iloc[0], tel['Y'].iloc[0]),
                textcoords='offset points', xytext=(12, 8),
                color=style.MUTED, fontsize=9, zorder=6)

def draw_corners(ax, session, fontsize=8):
    """Number each corner on a track map."""
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        return

    for index, (_, corner) in enumerate(corners.iterrows()):
        ax.plot(corner['X'], corner['Y'], marker='o', markersize=3,
                color=style.MUTED, zorder=5)
        offset = (8, 8) if index % 2 == 0 else (-14, -12)
        ax.annotate(f"{int(corner['Number'])}",
                    (corner['X'], corner['Y']),
                    textcoords='offset points', xytext=offset,
                    color=style.MUTED, fontsize=fontsize, zorder=5)

def mark_corners_on_axes(axes, session, label_axis=None):
    """Vertical corner lines on distance-based plots."""
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        return

    for _, corner in corners.iterrows():
        for axis in axes:
            axis.axvline(corner['Distance'], color=style.GRID,
                         linewidth=0.6, zorder=0)
        if label_axis is not None:
            label_axis.annotate(f"{int(corner['Number'])}",
                                (corner['Distance'], label_axis.get_ylim()[1]),
                                color=style.MUTED, fontsize=8,
                                ha='center', va='top')

def plot_lap(year, race, driver, lap_number=None, axes=None):
    """Speed, throttle and brake traces. Needs three stacked axes."""
    lap, tel, session = get_lap_telemetry(year, race, driver, lap_number)

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    else:
        fig = axes[0].get_figure()

    axes[0].plot(tel['Distance'], tel['Speed'], color=style.DRIVER_A)
    axes[0].set_ylabel('Speed (km/h)')

    axes[1].plot(tel['Distance'], tel['Throttle'], color=style.DRIVER_A)
    axes[1].set_ylabel('Throttle (%)')

    axes[2].fill_between(tel['Distance'], tel['Brake'].astype(int),
                         color=style.ACCENT, alpha=0.7, step='pre')
    axes[2].set_ylabel('Brake')
    axes[2].set_yticks([0, 1])
    axes[2].set_xlabel('Distance (m)')

    mark_corners_on_axes(axes, session, label_axis=axes[0])

    if standalone:
        style.title(fig, f"{driver} — {race} {year}",
                    f"Lap {int(lap['LapNumber'])}  ·  "
                    f"{format_lap_time(lap['LapTime'])}")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        display()
    else:
        axes[0].set_title(f"{driver} lap {int(lap['LapNumber'])}",
                          color=style.TEXT, fontsize=11)

def compare_laps(year, race, driver_a, driver_b, session_type='R', axes=None):
    """Speed traces with the time delta underneath. Needs two axes."""
    lap_a, tel_a, session = get_lap_telemetry(year, race, driver_a,
                                              session_type=session_type)
    lap_b, tel_b, _ = get_lap_telemetry(year, race, driver_b,
                                        session_type=session_type)

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig = axes[0].get_figure()

    max_distance = min(tel_a['Distance'].max(), tel_b['Distance'].max())
    grid = np.linspace(0, max_distance, 1000)

    def elapsed(tel):
        return (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds()

    time_a = np.interp(grid, tel_a['Distance'], elapsed(tel_a))
    time_b = np.interp(grid, tel_b['Distance'], elapsed(tel_b))
    delta = time_a - time_b

    speed_a = np.interp(grid, tel_a['Distance'], tel_a['Speed'])
    speed_b = np.interp(grid, tel_b['Distance'], tel_b['Speed'])

    axes[0].plot(grid, speed_a, label=driver_a, color=style.DRIVER_A)
    axes[0].plot(grid, speed_b, label=driver_b, color=style.DRIVER_B)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend(loc='lower right')

    axes[1].plot(grid, delta, color=style.TEXT, linewidth=1.5)
    axes[1].axhline(0, color=style.MUTED, linewidth=0.8)
    axes[1].fill_between(grid, delta, 0, where=delta > 0,
                         alpha=0.35, color=style.DRIVER_B)
    axes[1].fill_between(grid, delta, 0, where=delta < 0,
                         alpha=0.35, color=style.DRIVER_A)
    axes[1].set_ylabel('Delta (s)')
    axes[1].set_xlabel('Distance (m)')

    mark_corners_on_axes(axes, session, label_axis=axes[0])

    final = delta[-1]
    ahead = driver_b if final > 0 else driver_a

    if standalone:
        style.title(fig, f"{driver_a} vs {driver_b} — {race} {year}",
                    f"{driver_a} lap {int(lap_a['LapNumber'])} "
                    f"({format_lap_time(lap_a['LapTime'])})   vs   "
                    f"{driver_b} lap {int(lap_b['LapNumber'])} "
                    f"({format_lap_time(lap_b['LapTime'])})  ·  "
                    f"{ahead} quicker by {abs(final):.3f}s")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        display()
    else:
        axes[0].set_title(f"{driver_a} vs {driver_b}",
                          color=style.TEXT, fontsize=11)

def corner_analysis(year, race, driver_a, driver_b, session_type='R', ax=None):
    """Where each driver gains or loses, corner by corner."""
    lap_a, tel_a, session = get_lap_telemetry(year, race, driver_a,
                                              session_type=session_type)
    lap_b, tel_b, _ = get_lap_telemetry(year, race, driver_b,
                                        session_type=session_type)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 6))
    else:
        fig = ax.get_figure()

    corners = session.get_circuit_info().corners

    def elapsed(tel):
        return (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds()

    max_distance = min(tel_a['Distance'].max(), tel_b['Distance'].max())
    grid = np.linspace(0, max_distance, 2000)
    time_a = np.interp(grid, tel_a['Distance'], elapsed(tel_a))
    time_b = np.interp(grid, tel_b['Distance'], elapsed(tel_b))
    delta = time_a - time_b

    numbers, changes, speeds = [], [], []
    previous_distance = 0

    for _, corner in corners.iterrows():
        distance = corner['Distance']
        if distance > max_distance:
            continue
        before = np.interp(previous_distance, grid, delta)
        after = np.interp(distance, grid, delta)
        numbers.append(int(corner['Number']))
        changes.append(after - before)

        window = tel_a[(tel_a['Distance'] > distance - 100) &
                       (tel_a['Distance'] < distance + 100)]
        speeds.append(float(window['Speed'].min()) if len(window) else None)

        previous_distance = distance

    colours = [style.DRIVER_B if change > 0 else style.DRIVER_A
               for change in changes]

    ax.bar(range(len(numbers)), changes, color=colours)
    ax.axhline(0, color=style.MUTED, linewidth=0.8)
    ax.set_xticks(range(len(numbers)))
    ax.set_xticklabels([f"T{n}" for n in numbers], fontsize=9)
    ax.set_ylabel('Time change (s)')

    worst = numbers[int(np.argmax(changes))] if changes else None
    best = numbers[int(np.argmin(changes))] if changes else None

    if standalone:
        style.title(fig, f"Corner by corner — {driver_a} vs {driver_b}",
                    f"{race} {year}  ·  "
                    f"{driver_a} lap {int(lap_a['LapNumber'])} "
                    f"({format_lap_time(lap_a['LapTime'])})   vs   "
                    f"{driver_b} lap {int(lap_b['LapNumber'])} "
                    f"({format_lap_time(lap_b['LapTime'])})  ·  "
                    f"bars above zero = {driver_a} losing")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        display()
    else:
        ax.set_title(f"Corner delta — {driver_a} vs {driver_b}",
                     color=style.TEXT, fontsize=11)

    return {'corners': numbers, 'changes': changes, 'speeds': speeds,
            'worst': worst, 'best': best,
            'lap_a': int(lap_a['LapNumber']),
            'lap_b': int(lap_b['LapNumber']),
            'time_a': format_lap_time(lap_a['LapTime']),
            'time_b': format_lap_time(lap_b['LapTime'])}

def plot_speed_map(year, race, driver, lap_number=None, ax=None):
    """Track outline coloured by speed."""
    lap, tel, session = get_lap_telemetry(year, race, driver, lap_number)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 9))
    else:
        fig = ax.get_figure()

    points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ax.plot(tel['X'], tel['Y'], linewidth=11 if standalone else 6,
            color=style.TRACK, solid_capstyle='round', zorder=1)

    collection = LineCollection(segments, cmap='plasma',
                                linewidth=6 if standalone else 3,
                                capstyle='round', zorder=2)
    collection.set_array(tel['Speed'].to_numpy()[:-1])
    ax.add_collection(collection)

    draw_corners(ax, session, fontsize=8 if standalone else 6)
    if standalone:
        draw_start_line(ax, tel)

    ax.set_xlim(tel['X'].min() - 600, tel['X'].max() + 600)
    ax.set_ylim(tel['Y'].min() - 600, tel['Y'].max() + 600)
    ax.set_aspect('equal')
    ax.axis('off')

    if standalone:
        bar = fig.colorbar(collection, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label('Speed (km/h)', color=style.MUTED)
        bar.outline.set_edgecolor(style.GRID)
        bar.ax.tick_params(colors=style.MUTED)

        style.title(fig, f"{driver} — {race} {year}",
                    f"Lap {int(lap['LapNumber'])}  ·  "
                    f"{format_lap_time(lap['LapTime'])}"
                    f"  ·  top {tel['Speed'].max():.0f} km/h"
                    f"  ·  slowest {tel['Speed'].min():.0f} km/h")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        display()
    else:
        ax.set_title(f"{driver} speed, lap {int(lap['LapNumber'])}",
                     color=style.TEXT, fontsize=11)

def plot_gear_map(year, race, driver, lap_number=None, ax=None):
    """Track outline coloured by gear."""
    lap, tel, session = get_lap_telemetry(year, race, driver, lap_number)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 9))
    else:
        fig = ax.get_figure()

    points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ax.plot(tel['X'], tel['Y'], linewidth=11 if standalone else 6,
            color=style.TRACK, solid_capstyle='round', zorder=1)

    collection = LineCollection(segments, cmap='viridis',
                                linewidth=6 if standalone else 3,
                                capstyle='round', zorder=2)
    collection.set_array(tel['nGear'].to_numpy()[:-1].astype(float))
    ax.add_collection(collection)

    draw_corners(ax, session, fontsize=8 if standalone else 6)
    if standalone:
        draw_start_line(ax, tel)

    ax.set_xlim(tel['X'].min() - 600, tel['X'].max() + 600)
    ax.set_ylim(tel['Y'].min() - 600, tel['Y'].max() + 600)
    ax.set_aspect('equal')
    ax.axis('off')

    if standalone:
        bar = fig.colorbar(collection, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label('Gear', color=style.MUTED)
        bar.outline.set_edgecolor(style.GRID)
        bar.ax.tick_params(colors=style.MUTED)

        style.title(fig, f"{driver} — {race} {year}",
                    f"Gear selection  ·  lap {int(lap['LapNumber'])}")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        display()
    else:
        ax.set_title(f"{driver} gears, lap {int(lap['LapNumber'])}",
                     color=style.TEXT, fontsize=11)

def plot_strategy(year, race, ax=None):
    """Every driver's tyre stints as coloured bars."""
    session = load_session(year, race)
    laps = session.laps
    order = session.results['Abbreviation'].tolist()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 9))
    else:
        fig = ax.get_figure()

    ax.grid(False)
    used = set()

    for row, driver in enumerate(order):
        driver_laps = laps[laps['Driver'] == driver]
        if driver_laps.empty:
            continue

        start = 0
        for stint, stint_laps in driver_laps.groupby('Stint'):
            compound = str(stint_laps['Compound'].iloc[0])
            used.add(compound)
            length = len(stint_laps)
            ax.barh(row, length, left=start, height=0.65,
                    color=style.COMPOUND_COLOURS.get(
                        compound, style.COMPOUND_COLOURS['UNKNOWN']),
                    edgecolor=style.BACKGROUND, linewidth=1.5)
            start += length

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10 if standalone else 7)
    ax.invert_yaxis()
    ax.set_xlabel('Lap')
    ax.set_xlim(0, None)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        handles = [plt.Rectangle((0, 0), 1, 1, color=style.COMPOUND_COLOURS[c])
                   for c in style.COMPOUND_COLOURS if c in used]
        labels = [c.title() for c in style.COMPOUND_COLOURS if c in used]
        ax.legend(handles, labels, loc='lower right', ncols=len(labels))

        style.title(fig, f"Tyre strategy — {race} {year}",
                    "Stint length by compound, in finishing order")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        display()
    else:
        ax.set_title("Tyre strategy", color=style.TEXT, fontsize=11)

def plot_positions(year, race, ax=None):
    """Track position lap by lap for every driver."""
    session = load_session(year, race)
    laps = session.laps

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 8))
    else:
        fig = ax.get_figure()

    for driver in session.results['Abbreviation']:
        driver_laps = laps[laps['Driver'] == driver]
        if driver_laps.empty:
            continue

        team = driver_laps['Team'].iloc[0]
        colour = style.TEAM_COLOURS.get(team, style.MUTED)

        ax.plot(driver_laps['LapNumber'], driver_laps['Position'],
                color=colour, linewidth=1.8 if standalone else 1.2)

        if standalone:
            final = driver_laps['Position'].dropna()
            if len(final):
                ax.annotate(driver, (driver_laps['LapNumber'].iloc[-1],
                                     final.iloc[-1]),
                            textcoords='offset points', xytext=(6, -3),
                            color=colour, fontsize=9)

    ax.invert_yaxis()
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.set_yticks(range(1, 21, 2))

    if standalone:
        style.title(fig, f"Race positions — {race} {year}",
                    "Every driver's track position, lap by lap")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        display()
    else:
        ax.set_title("Positions", color=style.TEXT, fontsize=11)

def plot_gap_to_leader(year, race, drivers=None, ax=None):
    """Cumulative time gap to the race winner."""
    session = load_session(year, race)
    laps = session.laps

    if drivers is None:
        drivers = session.results['Abbreviation'].head(6).tolist()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 8))
    else:
        fig = ax.get_figure()

    winner = session.results['Abbreviation'].iloc[0]
    winner_laps = laps[laps['Driver'] == winner]
    winner_cumulative = winner_laps['LapTime'].dt.total_seconds().cumsum()
    winner_index = winner_laps['LapNumber'].to_numpy()

    for driver in drivers:
        driver_laps = laps[laps['Driver'] == driver]
        if driver_laps.empty:
            continue

        cumulative = driver_laps['LapTime'].dt.total_seconds().cumsum()
        reference = np.interp(driver_laps['LapNumber'].to_numpy(),
                              winner_index, winner_cumulative)
        gap = cumulative.to_numpy() - reference

        team = driver_laps['Team'].iloc[0]
        colour = style.TEAM_COLOURS.get(team, style.MUTED)
        ax.plot(driver_laps['LapNumber'], gap, label=driver,
                color=colour, linewidth=1.8 if standalone else 1.2)

    ax.axhline(0, color=style.MUTED, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Lap')
    ax.set_ylabel(f'Gap to {winner} (s)')
    ax.legend(loc='lower left', ncols=3, fontsize=9 if standalone else 7)

    if standalone:
        style.title(fig, f"Gap to leader — {race} {year}",
                    f"Cumulative time behind {winner}")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        display()
    else:
        ax.set_title(f"Gap to {winner}", color=style.TEXT, fontsize=11)

def animate_head_to_head(year, race, driver_a, driver_b, frames=500, ax=None):
    """Two fastest laps replayed together on the track map."""
    lap_a, tel_a, session = get_lap_telemetry(year, race, driver_a)
    lap_b, tel_b, _ = get_lap_telemetry(year, race, driver_b)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 9))
    else:
        fig = ax.get_figure()

    def elapsed(tel):
        return (tel['Time'] - tel['Time'].iloc[0]).dt.total_seconds()

    duration = max(elapsed(tel_a).max(), elapsed(tel_b).max())
    timeline = np.linspace(0, duration, frames)

    def track_position(tel):
        seconds = elapsed(tel)
        return (np.interp(timeline, seconds, tel['X']),
                np.interp(timeline, seconds, tel['Y']),
                np.interp(timeline, seconds, tel['Distance']),
                np.interp(timeline, seconds, tel['Speed']))

    xa, ya, da, sa = track_position(tel_a)
    xb, yb, db, sb = track_position(tel_b)

    ax.plot(tel_a['X'], tel_a['Y'], linewidth=11 if standalone else 7,
            color=style.TRACK, solid_capstyle='round', zorder=1)
    draw_corners(ax, session, fontsize=8 if standalone else 6)

    if standalone:
        draw_start_line(ax, tel_a)
        ax.set_xlim(tel_a['X'].min() - 8000, tel_a['X'].max() + 800)
    else:
        ax.set_xlim(tel_a['X'].min() - 800, tel_a['X'].max() + 800)

    ax.set_ylim(tel_a['Y'].min() - 800, tel_a['Y'].max() + 800)
    ax.set_aspect('equal')
    ax.axis('off')

    trail_a, = ax.plot([], [], linewidth=3, color=style.DRIVER_A,
                       alpha=0.5, zorder=2)
    trail_b, = ax.plot([], [], linewidth=3, color=style.DRIVER_B,
                       alpha=0.5, zorder=2)

    marker = 15 if standalone else 10
    car_b, = ax.plot([], [], 'o', markersize=marker, color=style.DRIVER_B,
                     markeredgecolor=style.BACKGROUND, markeredgewidth=2,
                     zorder=4)
    car_a, = ax.plot([], [], 'o', markersize=marker, color=style.DRIVER_A,
                     markeredgecolor=style.BACKGROUND, markeredgewidth=2,
                     alpha=0.85, zorder=5)

    text_size = 12 if standalone else 9
    clock = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    color=style.TEXT, fontsize=text_size + 1, va='top',
                    family='monospace', zorder=6)
    line_a = ax.text(0.02, 0.93, '', transform=ax.transAxes,
                     color=style.DRIVER_A, fontsize=text_size, va='top',
                     family='monospace', zorder=6)
    line_b = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                     color=style.DRIVER_B, fontsize=text_size, va='top',
                     family='monospace', zorder=6)
    gap_text = ax.text(0.02, 0.86, '', transform=ax.transAxes,
                       color=style.MUTED, fontsize=text_size, va='top',
                       family='monospace', zorder=6)

    if standalone:
        style.title(fig, f"{driver_a} vs {driver_b} — {race} {year}",
                    f"{driver_a} lap {int(lap_a['LapNumber'])} "
                    f"({format_lap_time(lap_a['LapTime'])})   vs   "
                    f"{driver_b} lap {int(lap_b['LapNumber'])} "
                    f"({format_lap_time(lap_b['LapTime'])})  ·  "
                    f"started together")
    else:
        ax.set_title(f"{driver_a} vs {driver_b}",
                     color=style.TEXT, fontsize=11)

    def update(frame):
        car_a.set_data([xa[frame]], [ya[frame]])
        car_b.set_data([xb[frame]], [yb[frame]])

        tail = max(0, frame - 25)
        trail_a.set_data(xa[tail:frame + 1], ya[tail:frame + 1])
        trail_b.set_data(xb[tail:frame + 1], yb[tail:frame + 1])

        clock.set_text(f"{timeline[frame]:5.1f}s")
        line_a.set_text(f"● {driver_a}  {sa[frame]:5.0f} km/h")
        line_b.set_text(f"● {driver_b}  {sb[frame]:5.0f} km/h")

        gap = da[frame] - db[frame]
        leader = driver_a if gap > 0 else driver_b
        gap_text.set_text(f"  {leader} ahead by {abs(gap):5.1f}m")

        return (car_a, car_b, trail_a, trail_b,
                clock, line_a, line_b, gap_text)

    animation = FuncAnimation(fig, update, frames=frames,
                              interval=25, blit=standalone, repeat=True)

    fig._animation = animation

    if standalone:
        display()

    return animation

if __name__ == '__main__':
    animate_head_to_head(2024, 'Monza', 'NOR', 'PIA')
    plt.show()