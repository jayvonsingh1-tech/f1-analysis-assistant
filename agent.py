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

load_dotenv()
fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
MODEL = "claude-haiku-4-5"

FUEL_EFFECT = 0.03      # seconds per lap per kg of fuel
FUEL_START = 100.0      # kg at race start (estimate)

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

def clean_laps(session, driver):
    laps = session.laps.pick_drivers(driver).copy()
    laps['Seconds'] = laps['LapTime'].dt.total_seconds()

    burn_per_lap = FUEL_START / session.total_laps
    fuel_remaining = FUEL_START - (laps['LapNumber'] - 1) * burn_per_lap
    laps['FuelCorrected'] = laps['Seconds'] - fuel_remaining * FUEL_EFFECT

    return laps

def analyse_stints(year, race, driver):
    session = load_race(year, race)
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
                     f"{len(stint_laps)} laps, avg {format_time(clean['Seconds'].mean())}, "
                     f"trend {slope:+.3f}s/lap ({label}){note}")

    lines.append("Note: trends are linear fits on green-flag, fuel-corrected "
                 "laps (assumes 100kg start, 0.03s/kg). Driver input cannot be "
                 "separated from compound or circuit effects. This tool does "
                 "not return finishing positions or gaps between drivers.")

    return "\n".join(lines)

def compare(year, race, driver_a, driver_b):
    session = load_race(year, race)

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
        if event_date < today:
            status = "completed"
            completed += 1
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
        lines.append(f"  Air {weather['AirTemp'].min():.1f}-{weather['AirTemp'].max():.1f}C, "
                     f"track {weather['TrackTemp'].min():.1f}-{weather['TrackTemp'].max():.1f}C, "
                     f"humidity {weather['Humidity'].mean():.0f}%")
        if weather['Rainfall'].any():
            lines.append(f"  Rain recorded on {weather['Rainfall'].mean() * 100:.0f}% "
                         f"of weather readings")
        else:
            lines.append("  Dry throughout")

    SKIP = (
        'WAVED BLUE FLAG',
        'IN TRACK SECTOR',
        'TRACK LIMITS',
        'ALL PASS HOLDERS',
        'AWNINGS',
        'PIT EXIT OPEN',
        'PIT EXIT CLOSED',
        'GREEN LIGHT - PIT EXIT',
        'OVERTAKE ENABLED',
        'OVERTAKE DISABLED',
        'TRACK CLEAR',
        'RISK OF RAIN',
        'GRIP DELTA ACTIVE',
        # Investigations that ended in no action are non-events
        'NO FURTHER INVESTIGATION',
        'NO FURTHER ACTION',
    )

    # Intermediate steps - keep only messages that changed something
    LIFECYCLE = (
        'NOTED -',
        'UNDER INVESTIGATION',
        'WILL BE INVESTIGATED',
    )
    OUTCOME = (
        'PENALTY',
        'REPRIMAND',
        'DISQUALIFIED',
        'BLACK AND WHITE FLAG',
    )

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

TOOLS = [
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
                "year": {"type": "integer", "description": "Season year, e.g. 2024"},
                "race": {"type": "string", "description": "Circuit name, e.g. Monza"},
                "driver": {"type": "string", "description": "Three-letter code, e.g. NOR"},
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
            "finishing positions or the gap at the flag. Use when the user asks "
            "who was faster, or asks about two drivers together."
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
            "Get the finishing order for a race: positions, driver codes and "
            "full names, teams, status, the winner's total race time, each "
            "other driver's gap to the winner in seconds, and a summary count "
            "of how many drivers finished, were lapped and retired. This is the "
            "only tool that returns finishing gaps — use it whenever the user "
            "asks who won, where someone finished, how far apart drivers were "
            "at the flag, or how many drivers finished."
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
        "name": "qualifying",
        "description": (
            "Get qualifying results for a race: grid order, driver codes and "
            "full names, each driver's best qualifying lap and their gap to "
            "pole. Use for questions about qualifying, grid positions, pole "
            "position or one-lap pace."
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
            "List every round in a season: round number, event name, location, "
            "date, whether it has happened yet, and how many rounds are "
            "complete. Use this to find out which races exist in a season, "
            "which was the most recent race, what the next race is, or to "
            "resolve a vague reference like 'the last grand prix'."
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
            "Get what actually happened during a race: which laps ran under "
            "safety car, VSC, yellow or red flag and how many were disrupted; "
            "weather including air and track temperature, humidity and "
            "rainfall; and the significant race control messages — red flags, "
            "restarts, penalties, collisions. Routine chatter is filtered out "
            "and the number omitted is reported. Use whenever the user asks "
            "what happened in a race, why pace changed unexpectedly, whether "
            "there were safety cars or incidents, or what the conditions were. "
            "Also call this when stint data looks anomalous."
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
]

FUNCTIONS = {
    'analyse_stints': analyse_stints,
    'compare': compare,
    'race_result': race_result,
    'qualifying': qualifying,
    'season_calendar': season_calendar,
    'race_events': race_events,
}

SYSTEM_PROMPT = """You are an F1 race engineer's analysis assistant.

Speak naturally, the way an engineer would talk to a colleague. Full sentences,
not bullet-point reports. Conversational and plain. Keep answers under about
100 words unless asked for more detail, and avoid tables, headers and markdown
— your output may be read aloud.

You have tools that pull real timing data from F1 sessions. Use them whenever a
question needs actual numbers — never estimate or recall lap times from memory,
always fetch them.

Never refuse a request because you believe a date is in the future. Your
training data ends before today. Call the tool and report what comes back; if
no data exists, the tool will error and you can say so then.

If the user refers to a race vaguely — "the last one", "the most recent race",
"the next race" — call season_calendar to work out which race they mean rather
than asking them.

If you notice you're missing something the user asked for, and a tool can
supply it, call that tool. Don't tell the user what you don't have when you
could go and fetch it.

If a stint's pace looks anomalous — a very large trend, or a later stint much
slower than an earlier one — call race_events before explaining it. A safety
car, red flag or rain usually accounts for it.

Use driver names exactly as the tools give them. Never expand a three-letter
code into a name yourself.

Never do arithmetic on the numbers the tools give you — no totals, differences,
counts or averages of your own. The tools provide summary counts where they
matter. If you need a number that no tool gave you, say you don't have it.

When reading stint data:
- Trends are fuel-corrected and computed on green-flag laps only.
- A negative trend means pace improving; positive means degradation.
- The data is labelled (improving/degrading/flat) — trust the label.
- The stint count and lap totals are stated explicitly. Use those numbers.

Only state race results, finishing positions, gaps between drivers,
championship standings or incidents if a tool returned them. Never state a gap
or a margin unless a tool gave you that exact number. If you don't have it, say
you don't have it.

Don't attribute causes the data doesn't show. Don't call a performance
excellent, strong or well managed — those are judgements about a driver that a
degradation number cannot support. Describe what happened and how big it was.

"His hards were flat across eighteen laps, so no real drop-off there" is good.
"Excellent tyre management on the hards" is not — same observation, but the
second invents a cause.

You can answer general F1 and engineering questions directly from your own
knowledge: aerodynamics, tyre construction, power units, regulations, race
strategy. Be precise and use correct terminology.

If a question is ambiguous — an unclear driver, or a genuinely unknown season —
ask rather than guessing."""

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