import json
import logging
import os
import time
from datetime import date

import fastf1
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

import dashboard
import telemetry

load_dotenv()
fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
MODEL = "claude-haiku-4-5"

FUEL_EFFECT = 0.03
FUEL_START = 100.0

TRACK_STATUS = {
    '1': 'green',
    '2': 'yellow',
    '4': 'safety car',
    '5': 'red flag',
    '6': 'VSC',
    '7': 'VSC ending',
}

def format_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}:{remainder:06.3f}"

def load_race(year, race):
    session = fastf1.get_session(year, race, 'R')
    session.load()
    return session

def check_driver_ran(session, driver):
    """Return an explanation if the driver has no usable laps, else None."""
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        return f"{driver} has no lap data for this session."

    if laps['LapTime'].notna().sum() == 0:
        return f"{driver} completed no timed laps in this session."

    row = session.results[session.results['Abbreviation'] == driver]
    if not row.empty:
        status = str(row['Status'].iloc[0]).strip()
        # Very recent races have no status yet - don't treat that as a DNF
        if not status or status.lower() == 'nan':
            return None
        if status != 'Finished' and 'Lapped' not in status:
            return (f"{driver} did not finish this race (status: {status}), "
                    f"completing {len(laps)} laps. Race-long analysis is "
                    f"incomplete and telemetry comparison may fail.")
    return None

def session_drivers(year, race):
    session = load_race(year, race)
    lines = [f"{race} {year} - drivers"]
    for _, row in session.results.iterrows():
        lines.append(f"  {row['Abbreviation']} - {row['FullName']} "
                     f"({row['TeamName']})")
    return "\n".join(lines)

def clean_laps(session, driver):
    laps = session.laps.pick_drivers(driver).copy()
    laps['Seconds'] = laps['LapTime'].dt.total_seconds()

    burn_per_lap = FUEL_START / session.total_laps
    fuel_remaining = FUEL_START - (laps['LapNumber'] - 1) * burn_per_lap
    laps['FuelCorrected'] = laps['Seconds'] - fuel_remaining * FUEL_EFFECT

    return laps

def analyse_stints(year, race, driver):
    session = load_race(year, race)

    problem = check_driver_ran(session, driver)
    if problem:
        return problem

    laps = clean_laps(session, driver)

    lines = [f"{driver} - {race} {year}",
             f"Stints: {laps['Stint'].nunique()}",
             f"Total laps completed: {len(laps)}"]

    for stint, stint_laps in laps.groupby('Stint'):
        clean = stint_laps[stint_laps['PitInTime'].isna()
                           & stint_laps['PitOutTime'].isna()
                           & stint_laps['Seconds'].notna()
                           & (stint_laps['TrackStatus'] == '1')]

        non_green = len(stint_laps) - len(clean)

        if len(clean) < 3:
            lines.append(f"Stint {int(stint)}: {len(stint_laps)} laps, "
                         f"too few green-flag laps to fit a trend")
            continue

        slope = np.polyfit(clean['TyreLife'], clean['FuelCorrected'], 1)[0]

        if np.isnan(slope):
            lines.append(f"Stint {int(stint)}: {clean['Compound'].iloc[0]}, "
                         f"{len(stint_laps)} laps, trend could not be computed")
            continue

        if slope < -0.02:
            label = "improving"
        elif slope > 0.02:
            label = "degrading"
        else:
            label = "flat"

        note = f", {non_green} laps excluded (pit/non-green)" if non_green else ""

        lines.append(f"Stint {int(stint)}: {clean['Compound'].iloc[0]}, "
                     f"{len(stint_laps)} laps, avg "
                     f"{format_time(clean['Seconds'].mean())}, "
                     f"trend {slope:+.3f}s/lap ({label}){note}")

    lines.append("Note: trends are linear fits on green-flag, fuel-corrected "
                 "laps (assumes 100kg start, 0.03s/kg). Driver input cannot be "
                 "separated from compound or circuit effects. This tool does "
                 "not return finishing positions or gaps between drivers.")

    return "\n".join(lines)

def sector_times(year, race, driver_a, driver_b=None):
    session = load_race(year, race)

    def best_sectors(driver):
        laps = session.laps.pick_drivers(driver)
        clean = laps[laps['Deleted'].fillna(False) == False]
        return {
            's1': clean['Sector1Time'].min(),
            's2': clean['Sector2Time'].min(),
            's3': clean['Sector3Time'].min(),
            'best_lap': clean['LapTime'].min(),
        }

def show_sector_map(year, race, driver, panel='main'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)

    info = board.main_info if panel == 'main' else None
    if info is not None:
        board.clear_panel(info)

    board.animation, data = telemetry.animate_sector_map(
        year, race, driver, ax=ax, info_ax=info)
    board.show()

    lines = [f"Qualifying sector map for {driver}, {race} {year}, playing in "
             f"the {panel} panel. Lap {data['lap_number']}, "
             f"{data['lap_time']}."]
    for sector in data['sectors']:
        if sector['time'] is None:
            lines.append(f"  S{sector['number']}: no time")
            continue
        lines.append(f"  S{sector['number']}: {sector['time']}s "
                     f"({sector['status']}; session best "
                     f"{sector['session_best']}s)")
    return "\n".join(lines)

def describe(driver, data):
    parts = []
    theoretical = 0
    for key in ('s1', 's2', 's3'):
        value = data[key]
        if pd.isna(value):
            parts.append(f"{key.upper()}: no time")
            continue
        seconds = value.total_seconds()
        theoretical += seconds
        parts.append(f"{key.upper()}: {seconds:.3f}s")

    actual = data['best_lap']
    actual_text = format_time(actual.total_seconds()) if not pd.isna(actual) else "none"

    return (f"{driver} best sectors — " + ", ".join(parts) +
            f"\n  theoretical best lap {format_time(theoretical)}, "
            f"actual best {actual_text}")

    lines = [f"{race} {year} - best sector times"]
    a = best_sectors(driver_a)
    lines.append(describe(driver_a, a))

    if driver_b:
        b = best_sectors(driver_b)
        lines.append(describe(driver_b, b))

        lines.append(f"\n{driver_a} vs {driver_b} by sector:")
        for key in ('s1', 's2', 's3'):
            if pd.isna(a[key]) or pd.isna(b[key]):
                continue
            gap = a[key].total_seconds() - b[key].total_seconds()
            faster = driver_b if gap > 0 else driver_a
            lines.append(f"  {key.upper()}: {faster} quicker by "
                         f"{abs(gap):.3f}s")

    lines.append("\nSectors are each driver's personal best in that sector "
                 "across the race, possibly on different laps. The "
                 "theoretical best is those three summed — a lap nobody "
                 "actually drove.")

    return "\n".join(lines)

def compare(year, race, driver_a, driver_b):
    session = load_race(year, race)

    for driver in (driver_a, driver_b):
        problem = check_driver_ran(session, driver)
        if problem:
            return problem

    lines = []
    for driver in (driver_a, driver_b):
        lines.append(analyse_stints(year, race, driver))

    lines.append(f"\n{driver_a} vs {driver_b} - {race} {year}")

    results = {}
    for driver in (driver_a, driver_b):
        laps = clean_laps(session, driver)
        clean = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]
        results[driver] = {
            'median': clean['Seconds'].median(),
            'best': clean['Seconds'].min(),
            'stops': laps['PitInTime'].notna().sum(),
        }
        r = results[driver]
        lines.append(f"  {driver}: median {format_time(r['median'])}, "
                     f"best {format_time(r['best'])}, {r['stops']} stops")

    gap = results[driver_a]['median'] - results[driver_b]['median']
    quicker = driver_b if gap > 0 else driver_a
    lines.append(f"  {quicker} quicker by {abs(gap):.3f}s per lap on median pace")

    return "\n".join(lines)

def race_result(year, race):
    session = load_race(year, race)
    results = session.results

    lines = [f"{race} {year} - race result"]
    for _, row in results.iterrows():
        position = row['Position']
        if np.isnan(position):
            continue

        time_value = row['Time']
        if pd.isna(time_value):
            gap = ""
        elif int(position) == 1:
            gap = f" - winning time {time_value}"
        else:
            gap = f" - +{time_value.total_seconds():.3f}s"

        lines.append(f"  {int(position)}. {row['Abbreviation']} "
                     f"({row['FullName']}, {row['TeamName']}) - "
                     f"{row['Status']}{gap}")

    classified = int(results['Position'].notna().sum())
    finished = int((results['Status'] == 'Finished').sum())
    lapped = int(results['Status'].astype(str).str.contains('Lapped', na=False).sum())
    retired = classified - finished - lapped

    lines.append(f"\nSummary: {classified} classified, {finished} finished on "
                 f"the lead lap, {lapped} lapped, {retired} did not finish")

    return "\n".join(lines)

def qualifying(year, race):
    session = fastf1.get_session(year, race, 'Q')
    session.load()

    results = session.results
    lines = [f"{race} {year} - qualifying"]

    pole_time = None

    for _, row in results.iterrows():
        position = row['Position']
        if np.isnan(position):
            continue

        best = None
        for segment in ('Q3', 'Q2', 'Q1'):
            value = row[segment]
            if value is not None and not pd.isna(value):
                best = value.total_seconds()
                break

        if best is None:
            lines.append(f"  {int(position)}. {row['Abbreviation']} "
                         f"({row['FullName']}) - no time")
            continue

        if pole_time is None:
            pole_time = best
            gap = ""
        else:
            gap = f" (+{best - pole_time:.3f})"

        lines.append(f"  {int(position)}. {row['Abbreviation']} "
                     f"({row['FullName']}, {row['TeamName']}) "
                     f"{format_time(best)}{gap}")

    return "\n".join(lines)

def season_calendar(year):
    schedule = fastf1.get_event_schedule(year)
    today = pd.Timestamp(date.today())

    lines = [f"{year} season calendar"]
    completed = 0

    for _, row in schedule.iterrows():
        if row['RoundNumber'] == 0:
            continue

        event_date = row['EventDate']
        if event_date.date() < today.date():
            status = "completed"
            completed += 1
        elif event_date.date() == today.date():
            status = "today"
        else:
            status = "upcoming"

        lines.append(f"  Round {row['RoundNumber']}: {row['EventName']} "
                     f"({row['Location']}, {row['Country']}) - "
                     f"{event_date.date()} - {status}")

    lines.append(f"\nSummary: {completed} rounds completed so far this season")

    return "\n".join(lines)

def race_events(year, race):
    session = load_race(year, race)
    lines = [f"{race} {year} - key events and conditions"]

    laps = session.laps
    status_by_lap = laps.groupby('LapNumber')['TrackStatus'].agg(
        lambda values: values.mode().iloc[0] if len(values.mode()) else '1')

    def describe(code):
        found = [name for digit, name in TRACK_STATUS.items() if digit in str(code)]
        return ", ".join(found) if found else f"status {code}"

    disrupted = [(int(lap), describe(status))
                 for lap, status in status_by_lap.items() if status != '1']

    lines.append("\nDisrupted laps:")
    if disrupted:
        for lap, description in disrupted:
            lines.append(f"  Lap {lap}: {description}")
        lines.append(f"  ({len(disrupted)} laps disrupted in total)")
    else:
        lines.append("  None - green throughout")

    weather = session.weather_data
    if weather is not None and len(weather):
        lines.append("\nConditions:")
        lines.append(f"  Air {weather['AirTemp'].min():.1f}-"
                     f"{weather['AirTemp'].max():.1f}C, "
                     f"track {weather['TrackTemp'].min():.1f}-"
                     f"{weather['TrackTemp'].max():.1f}C, "
                     f"humidity {weather['Humidity'].mean():.0f}%")
        if weather['Rainfall'].any():
            lines.append(f"  Rain recorded on "
                         f"{weather['Rainfall'].mean() * 100:.0f}% "
                         f"of weather readings")
        else:
            lines.append("  Dry throughout")

    SKIP = (
        'WAVED BLUE FLAG', 'IN TRACK SECTOR', 'TRACK LIMITS',
        'ALL PASS HOLDERS', 'AWNINGS', 'PIT EXIT OPEN', 'PIT EXIT CLOSED',
        'GREEN LIGHT - PIT EXIT', 'OVERTAKE ENABLED', 'OVERTAKE DISABLED',
        'TRACK CLEAR', 'RISK OF RAIN', 'GRIP DELTA ACTIVE',
        'NO FURTHER INVESTIGATION', 'NO FURTHER ACTION',
    )

    LIFECYCLE = ('NOTED -', 'UNDER INVESTIGATION', 'WILL BE INVESTIGATED')
    OUTCOME = ('PENALTY', 'REPRIMAND', 'DISQUALIFIED', 'BLACK AND WHITE FLAG')

    messages = session.race_control_messages
    if messages is not None and len(messages):
        kept = []
        dropped = 0
        no_action = 0
        seen = set()

        for _, row in messages.iterrows():
            text = str(row['Message'])
            upper = text.upper()

            if 'NO FURTHER INVESTIGATION' in upper or 'NO FURTHER ACTION' in upper:
                no_action += 1
                dropped += 1
                continue

            if any(skip in upper for skip in SKIP):
                dropped += 1
                continue

            if any(step in upper for step in LIFECYCLE) and \
               not any(end in upper for end in OUTCOME):
                dropped += 1
                continue

            if upper in seen:
                dropped += 1
                continue
            seen.add(upper)

            lap = row['Lap']
            lap_text = f"Lap {int(lap)}" if not pd.isna(lap) else "Pre-race"
            kept.append(f"  {lap_text}: {text}")

        lines.append("\nRace control (only events that changed the race):")
        lines.extend(kept)
        lines.append(f"  ({no_action} incidents were reviewed with no further "
                     f"action and are not listed. {dropped} routine or "
                     f"duplicate messages omitted in total.)")

    return "\n".join(lines)

def _corner_lines(result, driver_a, driver_b, race, year, panel):
    faster = driver_a if result['time_a'] < result['time_b'] else driver_b
    lines = [f"Displayed in the {panel} panel, {race} {year}. "
             f"{driver_a} lap {result['lap_a']} ({result['time_a']}) vs "
             f"{driver_b} lap {result['lap_b']} ({result['time_b']}). "
             f"{faster} set the quicker of the two laps. "
             f"These are each driver's single fastest race lap, not averages.",
             f"Per corner (positive = {driver_a} losing), with minimum "
             f"speed through that corner:"]
    for number, change, speed in zip(result['corners'], result['changes'],
                                     result['speeds']):
        speed_text = f", {speed:.0f} km/h min" if speed else ""
        lines.append(f"  T{number}: {change:+.3f}s{speed_text}")
    lines.append(f"Largest loss for {driver_a} at T{result['worst']}, "
                 f"largest gain at T{result['best']}.")
    return "\n".join(lines)

def show_head_to_head(year, race, driver_a, driver_b, panel='main'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)

    info = board.main_info if panel == 'main' else None
    if info is not None:
        board.clear_panel(info)

    # Corner chart first - drawing after the animation starts breaks blitting
    side3 = dashboard.panel_for('side3')
    board.clear_panel(side3)
    result = telemetry.corner_analysis(year, race, driver_a, driver_b,
                                       ax=side3)

    board.animation = telemetry.animate_head_to_head(
        year, race, driver_a, driver_b, ax=ax, info_ax=info)
    board.show()

    return ("Animation playing. Corner deltas shown in side3.\n"
            + _corner_lines(result, driver_a, driver_b, race, year, panel))

def show_corner_analysis(year, race, driver_a, driver_b, panel='side3'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    result = telemetry.corner_analysis(year, race, driver_a, driver_b, ax=ax)
    board.show()
    return _corner_lines(result, driver_a, driver_b, race, year, panel)

def show_speed_map(year, race, driver, panel='main'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    telemetry.plot_speed_map(year, race, driver, ax=ax)
    board.show()

    lap, tel, session = telemetry.get_lap_telemetry(year, race, driver)
    slowest = tel.loc[tel['Speed'].idxmin()]

    return (f"Speed map for {driver}, {race} {year}, lap "
            f"{int(lap['LapNumber'])} ({telemetry.format_lap_time(lap['LapTime'])}), "
            f"shown in the {panel} panel. Track coloured by speed with corner "
            f"numbers marked.\n"
            f"  Top speed {tel['Speed'].max():.0f} km/h\n"
            f"  Slowest point {tel['Speed'].min():.0f} km/h at "
            f"{slowest['Distance']:.0f}m into the lap\n"
            f"  Average speed {tel['Speed'].mean():.0f} km/h\n"
            f"  Full throttle for {(tel['Throttle'] > 95).mean() * 100:.0f}% "
            f"of the lap, on the brakes for "
            f"{tel['Brake'].mean() * 100:.0f}%")

def show_gear_map(year, race, driver, panel='main'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    telemetry.plot_gear_map(year, race, driver, ax=ax)
    board.show()

    lap, tel, session = telemetry.get_lap_telemetry(year, race, driver)
    gears = tel['nGear'].dropna().astype(int)
    counts = gears.value_counts().sort_index()

    lines = [f"Gear map for {driver}, {race} {year}, lap "
             f"{int(lap['LapNumber'])}, shown in the {panel} panel.",
             f"  Highest gear {gears.max()}, lowest {gears.min()}",
             f"  Time in each gear:"]
    for gear, count in counts.items():
        share = count / len(gears) * 100
        lines.append(f"    {gear}: {share:.0f}%")
    return "\n".join(lines)

def circuit_info(year, race):
    session = load_race(year, race)
    info = session.get_circuit_info()
    corners = info.corners

    lines = [f"{race} {year} - circuit layout",
             f"  {len(corners)} numbered corners"]

    for _, corner in corners.iterrows():
        letter = str(corner.get('Letter', '')).strip()
        label = f"{int(corner['Number'])}{letter}"
        lines.append(f"  Turn {label} at {corner['Distance']:.0f}m")

    if hasattr(info, 'marshal_sectors') and len(info.marshal_sectors):
        lines.append(f"  {len(info.marshal_sectors)} marshal sectors")

    lines.append("\nCorner names are not in this data. Refer to corners by "
                 "number only.")

    return "\n".join(lines)


def show_strategy(year, race, panel='side1'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    telemetry.plot_strategy(year, race, ax=ax)
    board.show()
    return (f"Tyre strategy for {race} {year} shown in the {panel} panel. "
            f"This tool returns no numeric data.")

def show_positions(year, race, panel='side2'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    telemetry.plot_positions(year, race, ax=ax)
    board.show()
    return (f"Position changes for {race} {year} shown in the {panel} panel. "
            f"This tool returns no numeric data.")

def show_gap_to_leader(year, race, panel='strip'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    telemetry.plot_gap_to_leader(year, race, ax=ax)
    board.show()
    return (f"Gap to leader for {race} {year} shown in the {panel} panel, "
            f"top six finishers. This tool returns no numeric data.")

def show_corner_technique(year, race, driver_a, driver_b, panel='side3'):
    board = dashboard.get_board(f"{race} {year}")
    ax = dashboard.panel_for(panel)
    board.clear_panel(ax)
    result = telemetry.corner_technique(year, race, driver_a, driver_b, ax=ax)
    board.show()

    lines = [f"Technique comparison, {driver_a} vs {driver_b}, {race} {year}. "
             f"{driver_a} lap {result['lap_a']} ({result['time_a']}) vs "
             f"{driver_b} lap {result['lap_b']} ({result['time_b']}). "
             f"Chart in the {panel} panel.",
             f"Corners where the gap exceeds {result['threshold']}s "
             f"(positive = {driver_a} losing):"]

    if not result['findings']:
        lines.append("  No corner exceeded the threshold — the laps were "
                     "very evenly matched.")
    else:
        for finding in result['findings']:
            lines.append(f"  T{finding['corner']}: {finding['change']:+.3f}s")
            if finding['min_speed_a'] and finding['min_speed_b']:
                lines.append(f"    apex speeds: {driver_a} "
                             f"{finding['min_speed_a']:.0f}, {driver_b} "
                             f"{finding['min_speed_b']:.0f} km/h")
            for note in finding['notes']:
                lines.append(f"    {note}")
            if not finding['notes']:
                lines.append("    no measurable difference in braking point, "
                             "apex speed, throttle point or exit speed")

    lines.append("\nThese are measurements of what differed, not explanations "
                 "of why. Braking point, apex speed, throttle application and "
                 "exit speed are measured; car setup, tyre state, fuel load, "
                 "traffic and dirty air are not in this data.")

    return "\n".join(lines)

def clear_dashboard():
    dashboard.reset()
    return "Dashboard cleared, all panels blank."

def close_visuals():
    telemetry.close_all()
    return "Closed all visual windows."

PANEL_PROPERTY = {
    "type": "string",
    "description": ("Where to display it: main, side1, side2, side3 or strip. "
                    "Optional - each visual has a sensible default."),
}

TOOLS = [
        {
        "name": "sector_times",
        "description": (
            "Get each driver's personal best sector times across a race or "
            "session, their theoretical best lap (the three best sectors "
            "summed), and their actual best lap. Pass driver_b to compare two "
            "drivers sector by sector. Use when the user asks about sectors, "
            "which part of the lap someone was strong in, or how much time a "
            "driver left on the table. Note the theoretical best is a lap "
            "nobody actually drove."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver_a": {"type": "string", "description": "Driver code"},
                "driver_b": {
                    "type": "string",
                    "description": "Optional second driver code, to compare",
                },
            },
            "required": ["year", "race", "driver_a"],
        },
    },
    {
        "name": "analyse_stints",
        "description": (
            "Get a stint-by-stint breakdown for one driver in one race. "
            "Returns the number of stints, total laps completed, and for each "
            "stint: tyre compound, stint length, average pace and the "
            "fuel-corrected degradation trend, computed on green-flag laps "
            "only. Does NOT return finishing position or gaps to other "
            "drivers. Use when the user asks about a single driver's race, "
            "their tyre strategy, or how their pace changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver": {"type": "string", "description": "Three-letter code"},
            },
            "required": ["year", "race", "driver"],
        },
    },
    {
        "name": "compare",
        "description": (
            "Compare two drivers in the same race. Returns both drivers' stint "
            "breakdowns plus median pace, best lap and pit stop counts, and "
            "identifies who was quicker on average lap time. Does NOT return "
            "finishing positions or the gap at the flag."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver_a": {"type": "string", "description": "First driver code"},
                "driver_b": {"type": "string", "description": "Second driver code"},
            },
            "required": ["year", "race", "driver_a", "driver_b"],
        },
    },
    {
        "name": "race_result",
        "description": (
            "Get the finishing order: positions, driver codes and full names, "
            "teams, status, the winner's race time, each driver's gap to the "
            "winner, and summary counts. This is the only tool that returns "
            "finishing gaps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
            },
            "required": ["year", "race"],
        },
    },
        {
        "name": "show_sector_map",
        "description": (
            "Display an animation of a driver's fastest qualifying lap, with "
            "each sector of the track colouring in as they complete it — "
            "purple for session best, green for personal best, yellow for "
            "slower. Use whenever the user asks about a driver's qualifying "
            "lap or qualifying pace. Returns no numeric data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver": {"type": "string", "description": "Three-letter code"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver"],
        },
    },
    {
        "name": "qualifying",
        "description": (
            "Get qualifying results: grid order, driver codes and full names, "
            "best qualifying lap and gap to pole."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
            },
            "required": ["year", "race"],
        },
    },
    {
        "name": "season_calendar",
        "description": (
            "List every round in a season with dates and whether it has "
            "happened. Use to resolve vague references like 'the last race'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
            },
            "required": ["year"],
        },
    },
    {
        "name": "race_events",
        "description": (
            "Get what happened during a race: safety cars, VSC, flags, weather, "
            "and the significant race control messages. Use when the user asks "
            "what happened, or when stint data looks anomalous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
            },
            "required": ["year", "race"],
        },
    },
    {
        "name": "show_head_to_head",
        "description": (
            "Display an animation of two drivers' fastest laps replayed on the "
            "track, AND return corner-by-corner time differences. Defaults to "
            "the main panel with corner deltas in side3."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver_a": {"type": "string", "description": "First driver code"},
                "driver_b": {"type": "string", "description": "Second driver code"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver_a", "driver_b"],
        },
    },
    {
        "name": "show_corner_analysis",
        "description": (
            "Display a chart AND return the numbers: how much time one driver "
            "gains or loses at each corner on their fastest laps, with corner "
            "minimum speeds. The only tool giving corner-level differences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver_a": {"type": "string", "description": "First driver code"},
                "driver_b": {"type": "string", "description": "Second driver code"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver_a", "driver_b"],
        },
    },
    {
        "name": "show_speed_map",
        "description": (
            "Display the circuit coloured by speed for one driver's fastest "
            "lap, with corner numbers. Returns no lap data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver": {"type": "string", "description": "Three-letter code"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver"],
        },
    },
    {
        "name": "show_gear_map",
        "description": (
            "Display the circuit coloured by gear selection for one driver's "
            "fastest lap. Returns no lap data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver": {"type": "string", "description": "Three-letter code"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver"],
        },
    },
    {
        "name": "show_strategy",
        "description": (
            "Display every driver's tyre stints as coloured bars. Returns no "
            "numeric data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race"],
        },
    },
    {
        "name": "show_positions",
        "description": (
            "Display every driver's track position lap by lap. Returns no "
            "numeric data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race"],
        },
    },
    {
        "name": "show_gap_to_leader",
        "description": (
            "Display the cumulative gap to the winner for the top six "
            "finishers. Returns no numeric data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race"],
        },
    },
    {
        "name": "clear_dashboard",
        "description": (
            "Blank every panel on the dashboard. Use when the user asks to "
            "clear it, start fresh, or see only one thing."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "close_visuals",
        "description": "Close the dashboard window entirely.",
        "input_schema": {"type": "object", "properties": {}},
    },
        {
        "name": "session_drivers",
        "description": (
            "List every driver in a race with their three-letter code, full "
            "name and team. Use this whenever you are not certain of a "
            "driver's code — never guess a code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
            },
            "required": ["year", "race"],
        },
    },
        {
        "name": "show_corner_technique",
        "description": (
            "The most detailed comparison available. For each corner where "
            "two drivers' times differ meaningfully, measures and reports the "
            "four inputs that explain corner-level time: braking point, apex "
            "speed, where throttle is reapplied, and exit speed. Use whenever "
            "the user asks WHY a driver is losing time, what they could do "
            "better, or how one driver's technique differs from another's. "
            "This is the only tool that explains a time difference rather "
            "than just measuring it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Season year"},
                "race": {"type": "string", "description": "Circuit name"},
                "driver_a": {"type": "string", "description": "The driver being assessed"},
                "driver_b": {"type": "string", "description": "The reference driver"},
                "panel": PANEL_PROPERTY,
            },
            "required": ["year", "race", "driver_a", "driver_b"],
        },
    },
]

FUNCTIONS = {
    'analyse_stints': analyse_stints,
    'compare': compare,
    'race_result': race_result,
    'qualifying': qualifying,
    'season_calendar': season_calendar,
    'race_events': race_events,
    'show_head_to_head': show_head_to_head,
    'show_corner_analysis': show_corner_analysis,
    'show_speed_map': show_speed_map,
    'show_gear_map': show_gear_map,
    'show_strategy': show_strategy,
    'show_positions': show_positions,
    'show_gap_to_leader': show_gap_to_leader,
    'clear_dashboard': clear_dashboard,
    'close_visuals': close_visuals,
   'session_drivers': session_drivers,
       'show_sector_map': show_sector_map,
}

SYSTEM_PROMPT = """You are an F1 race engineer's analysis assistant.

## Always show the visual

Visuals are not optional and you never need to be asked:
- Qualifying lap or qualifying pace → show_sector_map
- Any comparison of two drivers → show_head_to_head
- Where time is gained or lost on track → show_corner_analysis
Call these before answering. If the user asks for a specific panel, pass it.
If they want to see only one thing, call clear_dashboard first.

Panels: main (large, centre-left, with a readout beside it), side1, side2 and
side3 down the right, strip along the bottom.

## Ground everything in tool output

Never state a lap time, gap, margin, position, result or incident unless a
tool returned that exact number. If you don't have it, say so.

Never do arithmetic — no totals, differences, counts or averages of your own.
The tools provide the numbers that matter.

You cannot see the visuals you display. Never describe corner exits, driving
style, racing lines or how a lap looked.

If unsure of a driver's three-letter code, call session_drivers. Never guess.
Use names exactly as the tools give them.

Your training data ends before today, so never refuse a request because a date
looks like the future — call the tool and report what comes back. For vague
references like "the last race", call season_calendar.

## Reading the data

Stint trends are fuel-corrected, green-flag laps only. Negative means pace
improving, positive means degradation; the label states which, so trust it.

If a stint looks anomalous, call race_events before explaining it — a safety
car, red flag or rain usually accounts for it.

## Interpretation

Report what happened and how big it was. Don't attribute causes the data can't
show, and don't call a performance excellent or well managed — a degradation
number can't support that.

"His hards were flat across eighteen laps, so no real drop-off there" is good.
"Excellent tyre management on the hards" invents a cause.

If the user explicitly asks for your opinion or what someone should have done
differently, give a real one: start from the numbers, say it's your reading,
and note briefly at the end what you can't see — tyre allocation, fuel loads,
track position, team radio. Caveats go at the end and stay short; they don't
replace the answer.

## Voice

Speak like an engineer talking to a colleague. Full sentences, no bullet
points, no tables or markdown — your output may be read aloud. Under about 100
words unless more is asked for.

Numbers as figures: 0.026s/lap, not "zero point zero two six". Seconds are
seconds, never basis points or percentages.

Use proper terminology — undercut, overcut, thermal degradation, graining,
dirty air, traction zone, tyre working range, delta.

If no tool provides what the user asked for, say so. Never substitute a
different visual and describe it as if it were the one requested.

Never name a corner. You have corner numbers from the tools; use those. Corner
names from your own knowledge are frequently wrong.

## Explaining time loss

When the user asks why a driver is slower, what they could do better, or how
to improve, call show_corner_technique. It measures braking points, apex
speeds, throttle application and exit speeds.

Report what the measurements show. "He brakes 12m earlier and carries 6 km/h
less at the apex" is a finding. "He lacks confidence on the brakes" is not —
you cannot see confidence, downforce level, tyre state or fuel load. When the
user asks for advice, frame it as what the numbers point at, and name the
things you cannot see that might explain them instead.

You can answer general F1 and engineering questions from your own knowledge.
If a question is ambiguous, ask."""

def call_model(system, messages):
    for attempt in range(4):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=system,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as error:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            print(f"[API error: {error}. Retrying in {wait}s]")
            time.sleep(wait)

def run_agent():
    messages = []
    system = SYSTEM_PROMPT + f"\n\nToday's date is {date.today().isoformat()}."

    print("F1 assistant. Ask anything, or 'quit' to stop.\n")

    while True:
        user_input = input("> ").strip()
        if user_input in ('quit', 'exit', ''):
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            response = call_model(system, messages)

            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text":
                    print(f"\n{block.text}\n")

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"[running {block.name}({block.input})]")

                try:
                    result = FUNCTIONS[block.name](**block.input)
                except Exception as error:
                    result = f"Error: {error}"

                print(f"[result]\n{result}\n")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

run_agent()
