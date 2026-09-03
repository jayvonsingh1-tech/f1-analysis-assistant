# F1 Analysis Assistant

A natural-language tool for analysing Formula 1 race data. Ask a question in
plain English and get real telemetry analysis back.

## What it does

- Converts questions into function calls using an LLM
- Pulls real lap data via FastF1
- Computes stint pace, tyre degradation trends and head-to-head comparisons
- Explains the results in plain language

All numbers are computed in Python. The language model handles understanding
the question and describing the output — it never produces figures itself.

## Example

Asking `compare norris and piastri at monza 2024` returns:

```
NOR - Monza 2024
Stint 1: MEDIUM, 14 laps, avg 1:25.144, trend -0.169s/lap (improving)
Stint 2: HARD, 18 laps, avg 1:23.829, trend -0.055s/lap (improving)
Stint 3: HARD, 21 laps, avg 1:22.700, trend -0.056s/lap (improving)

NOR vs PIA - Monza 2024
  NOR: median 1:23.601, best 1:21.432, 2 stops
  PIA: median 1:23.480, best 1:21.943, 2 stops
  PIA quicker by 0.121s per lap on median pace
```

## Reading the output

A negative trend means lap times falling across the stint — usually fuel burn
and track evolution outweighing tyre wear. Positive means degradation.

## Setup

```
pip install fastf1 matplotlib numpy anthropic python-dotenv
```

Create a `.env` file containing your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Then run `python assistant.py`.

## Status

In development. Planned: weather data, conversational follow-ups, session
comparison, live timing.