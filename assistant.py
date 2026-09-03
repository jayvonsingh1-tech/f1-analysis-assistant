import json
import logging
import os

import fastf1
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
MODEL = "claude-haiku-4-5"

ROUTING_PROMPT = """You convert F1 questions into function calls.

Available functions:
- analyse_stints(year, race, driver) - stint breakdown for one driver
- compare(year, race, driver_a, driver_b) - head to head between two drivers

Drivers are three-letter codes: VER, NOR, PIA, LEC, HAM, RUS, SAI, ALO.
Races are circuit names: Monza, Silverstone, Spa, Barcelona, Monaco, Suzuka.

Respond with JSON only, no other text:
{"function": "name", "args": {...}}

If the question doesn't fit either function, respond:
{"function": null, "reason": "short explanation"}"""

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
    return laps

def analyse_stints(year, race, driver):
    session = load_race(year, race)
    laps = clean_laps(session, driver)

    lines = [f"{driver} - {race} {year}"]

    for stint, stint_laps in laps.groupby('Stint'):
        clean = stint_laps[stint_laps['PitInTime'].isna() & stint_laps['PitOutTime'].isna()]
        if len(clean) < 2:
            lines.append(f"Stint {int(stint)}: too few clean laps")
            continue

        slope = np.polyfit(clean['TyreLife'], clean['Seconds'], 1)[0]

        if slope < -0.02:
            label = "improving"
        elif slope > 0.02:
            label = "degrading"
        else:
            label = "flat"

        lines.append(f"Stint {int(stint)}: {clean['Compound'].iloc[0]}, "
                     f"{len(stint_laps)} laps, avg {format_time(clean['Seconds'].mean())}, "
                     f"trend {slope:+.3f}s/lap ({label})")

    return "\n".join(lines)

EXPLAIN_PROMPT = """You are an F1 race engineer. Given lap data, explain what
happened in 2-4 sentences. Be specific about tyre compounds, stint pace and
degradation trends. Negative trend means laps getting faster (fuel burn, track
evolution); positive means degradation. Do not invent anything not in the data."""

def explain(question, data):
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=EXPLAIN_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\nData:\n{data}"}]
    )
    return response.content[0].text

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

FUNCTIONS = {
    'analyse_stints': analyse_stints,
    'compare': compare,
}

def route(question):
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=ROUTING_PROMPT,
        messages=[{"role": "user", "content": question}]
    )
    text = response.content[0].text.strip()
    text = text.replace('```json', '').replace('```', '').strip()
    return json.loads(text)

while True:
    question = input("\n> ").strip()
    if question in ('quit', 'exit', ''):
        break

    try:
        plan = route(question)
        if plan['function'] is None:
            print(plan.get('reason', "Can't do that one"))
        else:
            data = FUNCTIONS[plan['function']](**plan['args'])
            print(data)
            print("\n" + explain(question, data))
    except Exception as error:
        print(f"Didn't work: {error}")
