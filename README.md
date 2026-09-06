# F1 Analysis Assistant

A natural-language interface to Formula 1 timing data. Ask a question in plain
English and get real telemetry analysis back, with visuals on a multi-panel
dashboard.

All numbers are computed in Python from official timing data. The language
model handles understanding the question, choosing which analysis to run, and
describing the result — it never produces figures itself.

![Speed map](speed-map.png)

## What it does

**Ask it things:**

- "How did Norris do at Monza last year?"
- "Compare Verstappen and Antonelli in the last race"
- "Where is Piastri losing time to Norris?"
- "What happened during the race?"
- "What should Verstappen have done differently?"

**It answers with:**

- Stint-by-stint pace and tyre degradation, fuel-corrected and filtered to
  green-flag laps
- Corner-by-corner time deltas between two drivers, with minimum speeds
- Sector times and theoretical best laps
- Race results, qualifying, grid positions and gaps
- Safety cars, red flags, penalties, weather and track conditions

**And shows it:**

- Animated head-to-head — two fastest laps replayed together on the circuit
- Qualifying sector map — sectors colour in as the car completes them, purple
  for session best, green for personal best, yellow for slower
- Circuit maps coloured by speed or gear, with corner numbers
- Tyre strategy across the whole grid
- Race position changes and gap-to-leader traces

## The analysis

**Fuel correction.** A car burns roughly 100kg of fuel over a race, and mass
costs about 0.03s per lap per kg. Raw lap times therefore improve through a
stint whether or not the tyres are holding up. Degradation trends here are
fitted to fuel-corrected times, so a negative trend means genuine improvement
rather than a lighter car.

The correction is a model, not a measurement — fuel loads are not published.
It assumes a 100kg start and linear burn, which is close enough to separate
fuel effect from tyre behaviour, but the tool states its assumptions in every
output rather than hiding them.

**Green-flag filtering.** Safety car, VSC and yellow-flag laps are excluded
from pace trends, along with in-laps and out-laps. Without this, a single
safety car period produces degradation figures that are physically impossible.

**Grounding.** Display tools return the underlying numbers alongside the
visual, and the model is instructed never to state a figure a tool didn't
give it. This matters more than it sounds: a language model asked to describe
a chart it cannot see will invent a plausible description.

## Setup

```
pip install fastf1 matplotlib numpy pandas anthropic python-dotenv PyQt5
```

Create a `.env` file with an Anthropic API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Then:

```
python agent.py
```

## Project structure

```
agent.py       Tool definitions, analysis functions, the agent loop
telemetry.py   All visualisations, each usable standalone or in a panel
dashboard.py   Multi-panel layout and shared board
style.py       Colours, fonts and matplotlib theme
live.py        Live timing scaffold (needs an F1TV subscription)
```

Every visual function takes an optional `ax` argument. Called without one it
opens its own window; called with one it draws into a dashboard panel. That
separation is what lets the agent compose a layout from the same functions
used for standalone analysis.

## Data

Timing data comes from [FastF1](https://github.com/theOehrly/Fast-F1), which
wraps the official F1 timing API. Available: lap and sector times, tyre
compound and age, pit timings, positions, weather, race control messages, and
telemetry at ~10Hz including speed, throttle, brake, gear, DRS and track
position.

Not available: fuel loads, tyre temperatures, brake temperatures, or any team
telemetry. Analysis that would need those is either modelled explicitly or not
attempted.

## Status

Working and in active development. Planned: engineering-level improvement
suggestions from telemetry, full race replay, quali-to-race pace delta,
teammate comparison across a season, and voice interaction.

Live timing during a session is scaffolded but requires an F1TV subscription
for stream authentication.