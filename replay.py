"""Race replay: position data for every driver across a full session.

The data layer is deliberately separate from rendering. race_positions()
returns plain arrays of where every car was and what was happening; the
renderer consumes that. Swapping matplotlib for a browser or 3D engine
later means rewriting only the renderer.

All times are seconds from the start of the race.
"""

import logging

import fastf1
import numpy as np
import pandas as pd

fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

def race_positions(year, race, resolution=1.0):
    """Build a common timeline with every driver's position on it.

    resolution: seconds between samples. 1.0 is smooth enough to watch and
    keeps a two-hour race to about 7000 frames.

    Returns a dict with:
      timeline   - array of seconds from race start
      drivers    - {code: {'x', 'y', 'present', 'team', 'lap',
                           'compound', 'position'}}
      track      - (x, y) arrays of the circuit outline
      events     - list of {'time', 'type', 'driver', 'text'}
      total_laps - int
    """
    session = fastf1.get_session(year, race, 'R')
    session.load()

    laps = session.laps
    codes = session.results['Abbreviation'].tolist()

    starts = laps['LapStartTime'].dropna()
    ends = laps['Time'].dropna()
    if starts.empty or ends.empty:
        raise ValueError("No usable lap timing for this session.")

    begin = starts.min().total_seconds()
    finish = ends.max().total_seconds()
    timeline = np.arange(0, finish - begin, resolution)

    drivers = {}
    track_x, track_y = None, None

    for code in codes:
        driver_laps = laps.pick_drivers(code)
        if driver_laps.empty:
            continue

        try:
            pos = driver_laps.get_pos_data()
        except Exception:
            continue

        if pos is None or pos.empty:
            continue

        seconds = pos['SessionTime'].dt.total_seconds().to_numpy() - begin
        x = pos['X'].to_numpy(dtype=float)
        y = pos['Y'].to_numpy(dtype=float)

        present = (timeline >= seconds.min()) & (timeline <= seconds.max())

        drivers[code] = {
            'x': np.interp(timeline, seconds, x),
            'y': np.interp(timeline, seconds, y),
            'present': present,
            'team': str(driver_laps['Team'].iloc[0]),
            'lap': _step_series(timeline, driver_laps, 'LapNumber', begin),
            'compound': _step_series(timeline, driver_laps, 'Compound',
                                     begin, numeric=False),
            'position': _step_series(timeline, driver_laps, 'Position', begin),
        }

        if track_x is None:
            track_x, track_y = x, y

    events = _build_events(session, laps, begin)

    return {
        'timeline': timeline,
        'drivers': drivers,
        'track': (track_x, track_y),
        'events': events,
        'total_laps': int(laps['LapNumber'].max()),
        'race_name': f"{race} {year}",
    }

def _step_series(timeline, driver_laps, column, begin, numeric=True):
    """Hold a per-lap value across the timeline until the next lap starts."""
    times = driver_laps['LapStartTime'].dt.total_seconds().to_numpy() - begin
    values = driver_laps[column].to_numpy()

    if numeric:
        out = np.full(len(timeline), np.nan)
    else:
        out = np.full(len(timeline), '', dtype=object)

    indices = np.searchsorted(times, timeline, side='right') - 1
    valid = indices >= 0
    out[valid] = values[indices[valid]]
    return out

def _build_events(session, laps, begin):
    """Pit stops, retirements and race control, timed from race start."""
    events = []

    stops = laps[laps['PitInTime'].notna()]
    for _, lap in stops.iterrows():
        events.append({
            'time': lap['PitInTime'].total_seconds() - begin,
            'type': 'pit',
            'driver': lap['Driver'],
            'text': f"{lap['Driver']} pits, lap {int(lap['LapNumber'])}",
        })

    finish_lap = laps['LapNumber'].max()
    for code in laps['Driver'].unique():
        driver_laps = laps.pick_drivers(code)
        last = driver_laps['LapNumber'].max()
        if last < finish_lap - 1:
            when = driver_laps['Time'].dropna()
            if len(when):
                events.append({
                    'time': when.max().total_seconds() - begin,
                    'type': 'retirement',
                    'driver': code,
                    'text': f"{code} out on lap {int(last)}",
                })

    KEEP = ('RED FLAG', 'SAFETY CAR', 'VSC DEPLOYED', 'CHEQUERED FLAG',
            'PENALTY', 'RACE START')

    messages = session.race_control_messages
    if messages is not None and len(messages):
        for _, row in messages.iterrows():
            text = str(row['Message']).upper()
            if not any(key in text for key in KEEP):
                continue

            when = row.get('Time')
            if pd.isna(when):
                continue

            try:
                offset = (pd.Timestamp(when)
                          - session.session_start_time).total_seconds()
            except Exception:
                continue

            events.append({
                'time': offset,
                'type': 'control',
                'driver': None,
                'text': str(row['Message']),
            })

    events.sort(key=lambda e: e['time'])
    return events

if __name__ == '__main__':
    data = race_positions(2024, 'Monza')
    print(f"Timeline: {len(data['timeline'])} samples, "
          f"{data['timeline'][-1]:.0f}s of racing")
    print(f"Drivers: {len(data['drivers'])}")
    print(f"Events: {len(data['events'])}")
    print(f"Total laps: {data['total_laps']}")
    for event in data['events'][:10]:
        print(f"  {event['time']:6.0f}s  {event['type']:12s} {event['text']}")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import style

def animate_race(data, speed=10, ax=None, info_ax=None, start_lap=None):
    """Play a full race back on the track map.

    speed: playback multiplier, 1 to 20. 10 means ten seconds of race per
    second of watching.
    start_lap: jump straight to this lap.
    """
    speed = max(1, min(20, speed))

    timeline = data['timeline']
    drivers = data['drivers']
    track_x, track_y = data['track']

    # Frame interval: 40ms of wall clock per frame at 25fps
    frame_seconds = speed / 25.0
    frame_times = np.arange(0, timeline[-1], frame_seconds)

    # Jump to a lap if asked
    offset = 0
    if start_lap is not None:
        leader_laps = next(iter(drivers.values()))['lap']
        matching = np.where(leader_laps >= start_lap)[0]
        if len(matching):
            offset = timeline[matching[0]]
            frame_times = frame_times[frame_times >= offset]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()

    ax.plot(track_x, track_y, linewidth=10 if standalone else 8,
            color=style.TRACK, solid_capstyle='round', zorder=1)
    ax.set_aspect('equal')
    ax.axis('off')

    margin = 400
    ax.set_xlim(np.nanmin(track_x) - margin, np.nanmax(track_x) + margin)
    ax.set_ylim(np.nanmin(track_y) - margin, np.nanmax(track_y) + margin)

    # One marker and one label per driver
    markers, labels = {}, {}
    for code, info in drivers.items():
        colour = style.TEAM_COLOURS.get(info['team'], style.MUTED)
        marker, = ax.plot([], [], 'o', markersize=10 if standalone else 8,
                          color=colour, markeredgecolor=style.BACKGROUND,
                          markeredgewidth=1.5, zorder=4)
        label = ax.text(0, 0, code, color=style.TEXT,
                        fontsize=7 if standalone else 6,
                        ha='center', va='center', zorder=5)
        markers[code] = marker
        labels[code] = label

    # Timing tower and ticker
    target = info_ax if info_ax is not None else ax
    if info_ax is not None:
        info_ax.clear()
        info_ax.axis('off')
        text_x, size = 0.05, 10
    else:
        text_x, size = 0.02, 8

    header = target.text(text_x, 0.98, '', transform=target.transAxes,
                         color=style.TEXT, fontsize=size + 3, va='top',
                         family='monospace', fontweight='bold', zorder=6)
    tower = target.text(text_x, 0.90, '', transform=target.transAxes,
                        color=style.TEXT, fontsize=size, va='top',
                        family='monospace', linespacing=1.5, zorder=6)
    ticker = target.text(text_x, 0.16, '', transform=target.transAxes,
                         color=style.MUTED, fontsize=size - 1, va='top',
                         family='monospace', linespacing=1.6, zorder=6)

    if standalone:
        style.title(fig, data['race_name'],
                    f"Race replay at {speed}x")

    def update(frame_index):
        now = frame_times[frame_index]
        sample = int(np.searchsorted(timeline, now))
        sample = min(sample, len(timeline) - 1)

        order = []

        for code, info in drivers.items():
            if not info['present'][sample]:
                markers[code].set_data([], [])
                labels[code].set_position((0, 0))
                labels[code].set_text('')
                continue

            x, y = info['x'][sample], info['y'][sample]
            markers[code].set_data([x], [y])
            labels[code].set_position((x, y))
            labels[code].set_text(code)

            position = info['position'][sample]
            if not np.isnan(position):
                order.append((position, code, info))

        order.sort()

        lap = 0
        if order:
            leader_lap = order[0][2]['lap'][sample]
            lap = 0 if np.isnan(leader_lap) else int(leader_lap)

        header.set_text(f"LAP {lap}/{data['total_laps']}")

        rows = []
        for position, code, info in order[:20]:
            compound = str(info['compound'][sample])[:1] or '-'
            rows.append(f"{int(position):2d}  {code}  {compound}")
        tower.set_text("\n".join(rows))

        recent = [e for e in data['events']
                  if now - 25 < e['time'] <= now]
        ticker.set_text("\n".join(e['text'] for e in recent[-4:]))

        artists = list(markers.values()) + list(labels.values())
        return artists + [header, tower, ticker]

    animation = FuncAnimation(fig, update, frames=len(frame_times),
                              interval=40, blit=False, repeat=True)
    fig._animation = animation

    if standalone:
        plt.show(block=False)
        plt.pause(0.1)

    return animation


if __name__ == '__main__':
    data = race_positions(2024, 'Monza')
    animate_race(data, speed=15)
    plt.show()