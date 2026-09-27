"""Desktop HUD wallpaper for the F1 Analysis Assistant.

Draws the championship, a circuit map traced from a real pole lap, the
countdown to the next race and the state of this repo as a HUD-style
wallpaper, using the same FastF1 cache and team colours as the rest of the
project. Then sets it as the Windows desktop background.

Usage:
    python desktop_hud.py              fetch fresh data, draw, set wallpaper
    python desktop_hud.py --no-set     draw only (writes hud_wallpaper.png)
    python desktop_hud.py --offline    skip the internet, reuse saved data
    python desktop_hud.py --install    redraw automatically every 30 minutes
    python desktop_hud.py --uninstall  stop the automatic redraw

From anywhere else in the project:
    import desktop_hud
    desktop_hud.update_in_background(status="Simulator: tyre model running")
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- Settings ----------------------------------------------------------------

PROJECT_NAME = 'F1 ANALYSIS ASSISTANT'
TOP_N = 6                       # drivers in the championship panel
REFRESH_MINUTES = 30            # how often --install redraws it
TASK_NAME = 'F1 Desktop HUD'

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / 'cache'      # the same FastF1 cache the rest of the project uses
FONT_DIR = HERE / 'assets' / 'fonts'

# Generated files live outside the project, so OneDrive isn't re-uploading a
# new image every 30 minutes and git never sees them.
# On Windows that's %LOCALAPPDATA%\F1DesktopHUD (paste it into Explorer's address bar).
STATE_DIR = Path(os.environ['LOCALAPPDATA']) / 'F1DesktopHUD' if os.environ.get('LOCALAPPDATA') else HERE
STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = STATE_DIR / 'hud_data.json'
OUTPUT = STATE_DIR / 'hud_wallpaper.png'
LOG_FILE = STATE_DIR / 'hud.log'

BASE_W, BASE_H = 2880, 1800     # layout is designed at this size, then scaled
SUPERSAMPLE = 2                 # draw at 2x and shrink, for smooth circles
RACE_LENGTH = timedelta(hours=2, minutes=30)

# --- Colours -----------------------------------------------------------------

def rgb(hex_colour):
    h = hex_colour.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

try:
    import style                # the project's own palette, so teams match
    TEAM_COLOURS = {team: rgb(c) for team, c in style.TEAM_COLOURS.items()}
    RED = rgb(style.ACCENT)
except Exception:
    TEAM_COLOURS = {}
    RED = rgb('#E10600')

CYAN = (70, 214, 255)
DEEP = (28, 92, 214)
ICE = (222, 246, 255)
MUTED = (112, 146, 168)
DIM = (38, 92, 120)
AMBER = (255, 176, 32)
BG_CENTRE = (9, 22, 34)
BG_EDGE = (3, 5, 9)

def team_colour(team):
    """Match an Ergast constructor name ('Red Bull', 'RB F1 Team'...) to style.py."""
    name = (team or '').lower()
    for key, colour in TEAM_COLOURS.items():
        if key.lower() in name or (name and name in key.lower()):
            return colour
    aliases = {'rb f1': 'Racing Bulls', 'visa': 'Racing Bulls', 'sauber': 'Kick Sauber'}
    for alias, key in aliases.items():
        if alias in name and key in TEAM_COLOURS:
            return TEAM_COLOURS[key]
    return CYAN

def speed_colour(t):
    """Slow corners deep blue, fast straights near-white."""
    if t < 0.5:
        a, b, t = DEEP, CYAN, t * 2
    else:
        a, b, t = CYAN, ICE, (t - 0.5) * 2
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))

# --- Small helpers -----------------------------------------------------------

def log(message):
    """Print, and keep the last 200 lines in hud.log (pythonw has no console)."""
    line = f"[{datetime.now():%d %b %H:%M}] {message}"
    print(line)
    try:
        old = LOG_FILE.read_text(encoding='utf-8').splitlines()[-199:] if LOG_FILE.exists() else []
        LOG_FILE.write_text('\n'.join(old + [line]) + '\n', encoding='utf-8')
    except OSError:
        pass

def load_saved():
    try:
        return json.loads(DATA_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}

def save(data):
    DATA_FILE.write_text(json.dumps(data, indent=1), encoding='utf-8')

def lap_time(delta):
    """pandas Timedelta -> 1:42.526"""
    seconds = delta.total_seconds()
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"

def points_text(points):
    return f"{points:g}"

def short_event(name):
    return name.upper().replace('GRAND PRIX', 'GP')

def ago(delta):
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return 'JUST NOW'
    if hours < 24:
        return f"{int(hours)} HR{'S' if hours >= 2 else ''} AGO"
    days = int(hours // 24)
    return f"{days} DAY{'S' if days > 1 else ''} AGO"

# --- Fetching (FastF1) -------------------------------------------------------

def driver_code(row):
    code = row.get('driverCode')
    if isinstance(code, str) and code:
        return code
    return str(row['familyName'])[:3].upper()

def get_standings(ergast):
    response = ergast.get_driver_standings(season='current', round='last')
    table = response.content[0]
    info = response.description.iloc[0]
    drivers = []
    for _, row in table.head(TOP_N).iterrows():
        teams = list(row['constructorNames'])
        drivers.append({
            'position': int(row['position']),
            'code': driver_code(row),
            'name': row['familyName'],
            'team': teams[-1] if teams else '',
            'points': float(row['points']),
        })
    return {'season': int(info['season']), 'round': int(info['round']), 'drivers': drivers}

def get_last_race(ergast):
    response = ergast.get_race_results(season='current', round='last')
    info = response.description.iloc[0]
    podium = [driver_code(row) for _, row in response.content[0].head(3).iterrows()]
    return {'season': int(info['season']), 'round': int(info['round']),
            'name': info['raceName'], 'podium': podium}

def race_start(event):
    """UTC start of the Grand Prix itself (not the sprint) within an event."""
    import pandas as pd
    for n in range(1, 6):
        if event.get(f'Session{n}') == 'Race':
            start = event.get(f'Session{n}DateUtc')
            if start is not None and not pd.isna(start):
                return pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc)
    return None

def get_calendar():
    """Round count, the start of the previous race and details of the next one."""
    import fastf1
    now = datetime.now(timezone.utc)
    info = {}
    for season in (now.year, now.year + 1):
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
        except Exception:
            break
        races = [(start, event) for _, event in schedule.iterrows()
                 if (start := race_start(event))]
        if season == now.year:
            info['season'] = season
            info['total_rounds'] = len(races)
            finished = [start for start, _ in races if start + RACE_LENGTH <= now]
            if finished:
                info['previous_start'] = finished[-1].isoformat()
        upcoming = [(start, event) for start, event in races if start + RACE_LENGTH > now]
        if upcoming:
            start, event = upcoming[0]
            info['next'] = {
                'season': season,
                'round': int(event['RoundNumber']),
                'name': event['EventName'],
                'location': event['Location'],
                'country': event['Country'],
                'start': start.isoformat(),
            }
            break
    return info

def load_track(year, round_number, label, key):
    """Circuit outline and speed trace from the pole-sitter's fastest lap."""
    import fastf1
    import pandas as pd
    session = fastf1.get_session(year, round_number, 'Q')
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    margin = None
    try:
        results = session.results
        code = results.iloc[0]['Abbreviation']
        lap = session.laps.pick_drivers(code).pick_fastest()
        gap = (results.iloc[1]['Q3'] - results.iloc[0]['Q3']).total_seconds()
        if not math.isnan(gap):
            margin = {'code': results.iloc[1]['Abbreviation'], 'gap': round(gap, 3)}
    except Exception:
        lap = session.laps.pick_fastest()
    if lap is None or pd.isna(lap['LapTime']):
        raise ValueError('no timed lap in this session')

    tel = lap.get_telemetry()
    xy = tel[['X', 'Y']].to_numpy(dtype=float)
    speed = tel['Speed'].to_numpy(dtype=float)
    keep = ~np.isnan(xy).any(axis=1) & ~np.isnan(speed)
    xy, speed = xy[keep], speed[keep]

    try:  # turn the map to the orientation F1 uses on TV
        angle = math.radians(session.get_circuit_info().rotation)
        xy = xy @ np.array([[math.cos(angle), math.sin(angle)],
                            [-math.sin(angle), math.cos(angle)]])
    except Exception:
        pass

    step = max(1, len(xy) // 900)
    event = session.event
    return {
        'key': key, 'label': label, 'year': year,
        'circuit': event['Location'], 'event': event['EventName'],
        'points': np.round(xy[::step]).astype(int).tolist(),
        'speed': np.round(speed[::step]).astype(int).tolist(),
        'pole': {'code': lap['Driver'], 'time': lap_time(lap['LapTime']),
                 'top_speed': int(speed.max())},
        'margin': margin,
    }

def get_track(data, saved):
    """Next race's circuit if it was raced last season, otherwise the last race's."""
    import fastf1
    options = []
    upcoming = (data.get('calendar') or {}).get('next')
    if upcoming:
        try:
            before = fastf1.get_event_schedule(upcoming['season'] - 1, include_testing=False)
            same = before[before['Location'].str.lower() == upcoming['location'].lower()]
            if len(same):
                options.append((upcoming['season'] - 1, int(same.iloc[0]['RoundNumber']),
                                'NEXT CIRCUIT'))
        except Exception as error:
            log(f"couldn't check last season's calendar: {error}")
    last = data.get('last_race')
    if last:
        options.append((last['season'], last['round'], 'LAST RACE'))

    old = saved.get('track') or {}
    for year, round_number, label in options:
        key = f'{year}-{round_number}'
        if old.get('key') == key:          # already traced, skip the big download
            return {**old, 'label': label}
        try:
            return load_track(year, round_number, label, key)
        except Exception as error:
            log(f"no usable pole lap for {year} round {round_number}: {error}")
    return old or None

def fetch_all(saved):
    """Fetch every panel's data. Anything that fails keeps its last saved copy,
    so the wallpaper still works on the bus with no signal."""
    import logging
    import fastf1
    from fastf1.ergast import Ergast

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.WARNING)
    ergast = Ergast()

    data = dict(saved)
    fresh = 0
    parts = {
        'standings': lambda: get_standings(ergast),
        'last_race': lambda: get_last_race(ergast),
        'calendar': get_calendar,
    }
    for name, getter in parts.items():
        try:
            data[name] = getter()
            fresh += 1
        except Exception as error:
            log(f"couldn't fetch {name} ({error}), keeping the saved copy")
    try:
        data['track'] = get_track(data, saved)
    except Exception as error:
        log(f"couldn't build the track map ({error})")
    if fresh:
        data['fetched_at'] = datetime.now(timezone.utc).isoformat()
    return data

def find_git():
    """git on PATH, or the copy bundled inside GitHub Desktop."""
    found = shutil.which('git')
    if found:
        return found
    local = os.environ.get('LOCALAPPDATA')
    if local:
        bundled = sorted(Path(local).glob('GitHubDesktop/app-*/resources/app/git/cmd/git.exe'))
        if bundled:
            return str(bundled[-1])
    return None

def repo_info():
    """Latest commit and uncommitted changes in this folder's git repo."""
    exe = find_git()
    if exe is None:
        return repo_info_from_files()
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)   # no console flash on Windows

    def git(*args):
        result = subprocess.run([exe, '-C', str(HERE), *args], capture_output=True,
                                text=True, timeout=5, creationflags=flags)
        return result.stdout.strip() if result.returncode == 0 else ''

    try:
        last = git('log', '-1', '--format=%ct|%s')
        remote = git('remote', 'get-url', 'origin')
        changed = git('status', '--porcelain')
    except (OSError, subprocess.SubprocessError):
        return repo_info_from_files()
    if not last:
        return repo_info_from_files()
    stamp, message = last.split('|', 1)
    name = remote.rstrip('/').split('/')[-1].removesuffix('.git') if remote else HERE.name
    return {'name': name, 'time': int(stamp), 'message': message,
            'uncommitted': len(changed.splitlines())}

def repo_info_from_files():
    """Last resort: the newest entry in .git/logs/HEAD (commit, pull, etc.)."""
    try:
        line = (HERE / '.git' / 'logs' / 'HEAD').read_text(encoding='utf-8').splitlines()[-1]
        head, _, message = line.partition('\t')
        stamp = int(head.split()[-2])
    except (OSError, IndexError, ValueError):
        return None
    return {'name': HERE.name, 'time': stamp, 'message': message,
            'uncommitted': 0}

# --- Drawing -----------------------------------------------------------------

FONTS = {  # bundled font, then Windows fallbacks
    'display': ('Orbitron[wght].ttf', 'Bold', ['bahnschrift.ttf', 'segoeuib.ttf']),
    'bold': ('Rajdhani-Bold.ttf', None, ['bahnschrift.ttf', 'segoeuib.ttf']),
    'semi': ('Rajdhani-SemiBold.ttf', None, ['bahnschrift.ttf', 'segoeui.ttf']),
    'mono': ('ShareTechMono-Regular.ttf', None, ['consola.ttf', 'cour.ttf']),
}

class Hud:
    """Two transparent layers: 'glow' gets a light bloom, 'flat' stays crisp.

    Positions are in design units on a 2880x1800 canvas, scaled to the real
    screen and drawn at SUPERSAMPLE x size so circles come out smooth.
    """

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.k = min(width / BASE_W, height / BASE_H)     # design unit -> screen px
        self.ss = self.k * SUPERSAMPLE                     # design unit -> layer px
        self.w, self.h = width / self.k, height / self.k   # canvas in design units
        self.dy = (self.h - BASE_H) / 2                    # centre on taller screens
        size = (width * SUPERSAMPLE, height * SUPERSAMPLE)
        self.layers = {'glow': Image.new('RGBA', size), 'flat': Image.new('RGBA', size)}
        self.pens = {name: ImageDraw.Draw(img) for name, img in self.layers.items()}
        self.fonts = {}

    def p(self, x, y):
        return (x * self.ss, (y + self.dy) * self.ss)

    def px(self, width):
        return max(1, round(width * self.ss))

    def line(self, points, c, width=2, alpha=255, layer='glow'):
        self.pens[layer].line([self.p(x, y) for x, y in points], fill=(*c, alpha),
                              width=self.px(width), joint='curve')

    def ring(self, cx, cy, r, c, width=2, alpha=255, start=0, end=360, layer='glow'):
        """Circle or arc centred on radius r. Angles: clockwise degrees from 12 o'clock."""
        r += width / 2
        box = [*self.p(cx - r, cy - r), *self.p(cx + r, cy + r)]
        if end - start >= 360:
            self.pens[layer].ellipse(box, outline=(*c, alpha), width=self.px(width))
        elif end > start:
            self.pens[layer].arc(box, start - 90, end - 90, fill=(*c, alpha),
                                 width=self.px(width))

    def tick(self, cx, cy, angle, r0, r1, c, width=2, alpha=255, layer='glow'):
        dx, dy = math.sin(math.radians(angle)), -math.cos(math.radians(angle))
        self.line([(cx + dx * r0, cy + dy * r0), (cx + dx * r1, cy + dy * r1)],
                  c, width, alpha, layer)

    def rect(self, x0, y0, x1, y1, c, alpha=255, layer='flat'):
        self.pens[layer].rectangle([*self.p(x0, y0), *self.p(x1, y1)], fill=(*c, alpha))

    def dot(self, x, y, r, c, alpha=255, layer='glow'):
        self.pens[layer].ellipse([*self.p(x - r, y - r), *self.p(x + r, y + r)], fill=(*c, alpha))

    def font(self, kind, size):
        if (kind, size) not in self.fonts:
            name, variation, fallbacks = FONTS[kind]
            pixels = max(6, round(size * self.ss))
            loaded = None
            for path in [FONT_DIR / name] + [Path('C:/Windows/Fonts') / f for f in fallbacks]:
                try:
                    loaded = ImageFont.truetype(str(path), pixels)
                    break
                except OSError:
                    continue
            loaded = loaded or ImageFont.load_default(pixels)
            if variation:
                try:
                    loaded.set_variation_by_name(variation)
                except Exception:
                    pass
            self.fonts[(kind, size)] = loaded
        return self.fonts[(kind, size)]

    def width_of(self, s, kind, size, spacing=0):
        font = self.font(kind, size)
        if not spacing:
            return font.getlength(s) / self.ss
        return (sum(font.getlength(ch) for ch in s) / self.ss) + spacing * max(0, len(s) - 1)

    def text(self, x, y, s, kind, size, c, alpha=255, anchor='ls', spacing=0,
             halo=0.0, layer='flat'):
        """anchor: first letter l/m/r (horizontal), second a/m/s/d (vertical).
        spacing adds letter-spacing; halo adds a soft glow behind the text."""
        font = self.font(kind, size)
        x -= {'l': 0, 'm': 0.5, 'r': 1}[anchor[0]] * self.width_of(s, kind, size, spacing)
        targets = [(layer, alpha)]
        if halo:
            targets.append(('glow', round(alpha * halo)))
        for target, a in targets:
            px, py = self.p(x, y)
            if not spacing:
                self.pens[target].text((px, py), s, font=font, fill=(*c, a), anchor='l' + anchor[1])
                continue
            for ch in s:
                self.pens[target].text((px, py), ch, font=font, fill=(*c, a), anchor='l' + anchor[1])
                px += font.getlength(ch) + spacing * self.ss

    def fit(self, s, kind, size, max_width, spacing=0):
        """Shorten s with '...' until it fits max_width design units."""
        if self.width_of(s, kind, size, spacing) <= max_width:
            return s
        while s and self.width_of(s + '...', kind, size, spacing) > max_width:
            s = s[:-1]
        return s.rstrip() + '...'

    def finish(self, background):
        """Shrink the layers, bloom the glow layer, stack everything on the background."""
        size = (self.width, self.height)
        glow = np.asarray(self.layers['glow'].resize(size, Image.LANCZOS), np.float32) / 255
        flat = np.asarray(self.layers['flat'].resize(size, Image.LANCZOS), np.float32) / 255
        light = Image.fromarray((glow[..., :3] * glow[..., 3:] * 255).astype(np.uint8))
        near = np.asarray(light.filter(ImageFilter.GaussianBlur(5 * self.k)), np.float32) / 255
        far = np.asarray(light.filter(ImageFilter.GaussianBlur(26 * self.k)), np.float32) / 255
        out = np.asarray(background, np.float32) / 255 + near * 0.85 + far * 0.6
        for layer in (glow, flat):
            alpha = layer[..., 3:]
            out = out * (1 - alpha) + layer[..., :3] * alpha
        return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))

def background(width, height, centre):
    """Dark radial gradient, lit behind the ring. The noise stops OLED banding."""
    cx, cy = centre
    ys = ((np.arange(height, dtype=np.float32) - cy) / width)[:, None]
    xs = ((np.arange(width, dtype=np.float32) - cx) / width)[None, :]
    t = (np.clip(np.sqrt(xs ** 2 + ys ** 2) / 0.55, 0, 1) ** 1.3)[..., None]
    img = np.array(BG_CENTRE, np.float32) * (1 - t) + np.array(BG_EDGE, np.float32) * t
    img += np.random.default_rng(7).uniform(-0.7, 0.7, img.shape).astype(np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

def panel(hud, x0, y0, x1, y1, title, note=''):
    """Faint box with corner brackets and a title tab."""
    hud.rect(x0, y0, x1, y1, CYAN, 8)
    arm = 26
    for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        hud.line([(x + sx * arm, y), (x, y), (x, y + sy * arm)], CYAN, 2.5, 230)
    hud.line([(x0 + arm + 16, y0), (x1 - arm - 16, y0)], CYAN, 1.2, 55)
    hud.rect(x0 + 30, y0 + 35, x0 + 41, y0 + 46, RED, 255, layer='glow')
    hud.text(x0 + 56, y0 + 52, title, 'bold', 28, CYAN, spacing=5)
    if note:
        hud.text(x1 - 30, y0 + 52, note, 'mono', 24, MUTED, anchor='rs')

def draw_backdrop(hud, left, right, cx, cy, R):
    for gx in np.arange(left - 20, hud.w, 60):          # dot grid
        for gy in np.arange(200, 1700, 60):
            hud.dot(gx, gy, 1.3, CYAN, 26, layer='flat')
    for i in range(90):                                  # faint dashed halo
        hud.ring(cx, cy, R + 120, CYAN, 1.2, 40, i * 4, i * 4 + 1.6, layer='flat')
    x = left + 10                                        # vertical scale on the left
    hud.line([(x, 300), (x, 1340)], CYAN, 1.2, 45, layer='flat')
    for i, y in enumerate(range(300, 1341, 26)):
        hud.line([(x, y), (x + (18 if i % 5 == 0 else 8), y)], CYAN, 1.2,
                 90 if i % 5 == 0 else 45, layer='flat')

def draw_header(hud, data, now, left, right):
    y = 112
    width = hud.width_of(PROJECT_NAME, 'bold', 34, 7)
    hud.text(left, y, PROJECT_NAME, 'bold', 34, ICE, spacing=7, halo=0.25)
    hud.text(left + width + 28, y, '// DESKTOP HUD', 'semi', 26, MUTED, spacing=4)

    local = now.astimezone()
    date_text = f"{local:%a %d %b %Y}".upper()
    hud.text(right, y, date_text, 'mono', 28, CYAN, anchor='rs')
    x = right - hud.width_of(date_text, 'mono', 28) - 36

    fetched = data.get('fetched_at')
    fetched = datetime.fromisoformat(fetched) if fetched else None
    if fetched and now - fetched < timedelta(hours=3):
        status, colour = f"DATA FASTF1  ·  SYNC {fetched.astimezone():%H:%M}", MUTED
    elif fetched:
        status, colour = f"OFFLINE  ·  DATA FROM {fetched.astimezone():%d %b %H:%M}".upper(), AMBER
    else:
        status, colour = 'NO DATA YET', AMBER
    hud.text(x, y, status, 'semi', 24, colour, anchor='rs', spacing=3)
    hud.dot(x - hud.width_of(status, 'semi', 24, 3) - 20, y - 9, 5,
            RED if colour is MUTED else AMBER)

    hud.line([(left, y + 30), (right, y + 30)], CYAN, 1.2, 60)
    hud.line([(left, y + 30), (left + 220, y + 30)], CYAN, 3, 230)
    for tx in np.arange(left, right + 1, 40):
        hud.line([(tx, y + 30), (tx, y + 38)], CYAN, 1.2, 55, layer='flat')

def draw_ring(hud, data, cx, cy, R, left, panel_x):
    for i in range(120):                                 # compass ticks
        major = i % 5 == 0
        hud.tick(cx, cy, i * 3, R + 8, R + (26 if major else 16), CYAN,
                 2 if major else 1.4, 170 if major else 70)
    hud.ring(cx, cy, R, CYAN, 1.5, 90)

    # Season progress: one segment per round, done ones lit, next one in red
    calendar = data.get('calendar') or {}
    total = calendar.get('total_rounds') or 0
    done = (data.get('standings') or {}).get('round') or 0
    r = R - 34
    if total:
        seg = 360 / total
        for i in range(total):
            a0, a1 = i * seg + 0.9, (i + 1) * seg - 0.9
            if i < done:
                hud.ring(cx, cy, r, CYAN, 10, 235, a0, a1)
            elif i == done:
                hud.ring(cx, cy, r, RED, 10, 235, a0, a1)
            else:
                hud.ring(cx, cy, r + 3, DIM, 4, 170, a0, a1)
        season = calendar.get('season') or (data.get('standings') or {}).get('season', '')
        caption = f"SEASON {season}  ·  ROUND {done} OF {total}"
        hud.text(cx, cy - R - 62, caption, 'semi', 26, MUTED, anchor='ms', spacing=6)

    for i in range(72):                                  # segmented ring
        hud.ring(cx, cy, R - 70, CYAN, 6, 50, i * 5, i * 5 + 3)
    for start in (30, 150, 270):                         # bracket arcs
        hud.ring(cx, cy, R - 96, CYAN, 3, 160, start, start + 60)
        hud.ring(cx, cy, R - 106, CYAN, 1.2, 80, start + 64, start + 100)
    hud.ring(cx, cy, R - 118, CYAN, 1.5, 60)
    for a in (0, 90, 180, 270):
        hud.tick(cx, cy, a, R - 118, R - 140, CYAN, 2, 150)

    # connectors out to the left scale and the right-hand panels
    hud.line([(cx + R + 40, cy), (panel_x - 30, cy)], CYAN, 1.2, 80)
    hud.dot(panel_x - 30, cy, 4, CYAN, 200)
    hud.line([(left + 40, cy), (cx - R - 40, cy)], CYAN, 1.2, 60)
    for x in np.arange(left + 40, cx - R - 39, 20):
        hud.line([(x, cy - 5), (x, cy + 5)], CYAN, 1.2, 60, layer='flat')

def draw_track(hud, track, cx, cy, R):
    if not track or not track.get('points'):
        hud.text(cx, cy, 'NO TRACK DATA YET', 'semi', 30, MUTED, anchor='mm', spacing=6)
        return
    pts = np.array(track['points'], float)
    pts[:, 1] *= -1                                      # screen y runs downwards
    centre = (pts.min(0) + pts.max(0)) / 2
    rel = pts - centre
    xy = rel * ((R - 150) / np.linalg.norm(rel, axis=1).max()) + (cx, cy)
    loop = [tuple(p) for p in xy] + [tuple(xy[0])]

    hud.line(loop, CYAN, 16, 45)                         # soft body under the line
    speed = np.array(track.get('speed') or [], float)
    if len(speed) == len(xy):
        t = (speed - speed.min()) / max(1.0, np.ptp(speed))
        t = np.append(t, t[0])
        for i in range(len(loop) - 1):
            hud.line([loop[i], loop[i + 1]], speed_colour((t[i] + t[i + 1]) / 2), 5, 255)
    else:
        hud.line(loop, ICE, 5, 245)

    d = xy[min(4, len(xy) - 1)] - xy[0]                  # start/finish marker
    d = d / (np.linalg.norm(d) or 1)
    n = np.array([-d[1], d[0]])
    a, b = xy[0] - n * 22, xy[0] + n * 22
    hud.line([tuple(a), tuple(b)], RED, 6, 255)
    lx, ly = xy[0] + n * 50
    hud.text(lx, ly, 'S/F', 'mono', 20, RED, anchor='mm')

    # Titles and readouts under the ring
    hud.text(cx, cy + R + 118, track['circuit'].upper(), 'display', 60, ICE,
             anchor='ms', spacing=14, halo=0.35)
    subtitle = f"{track['label']}  ·  {track['event']} {track['year']}".upper()
    hud.text(cx, cy + R + 168, subtitle, 'semi', 28, CYAN, 220, anchor='ms', spacing=5)

    readouts = []
    if track.get('pole'):
        readouts.append(('POLE LAP', f"{track['pole']['code']}  {track['pole']['time']}"))
    if track.get('margin'):
        readouts.append(('MARGIN', f"+{track['margin']['gap']:.3f}  {track['margin']['code']}"))
    if (track.get('pole') or {}).get('top_speed') and len(speed):
        readouts.append(('TOP SPEED', f"{track['pole']['top_speed']} KM/H"))
    spread = 330
    x = cx - spread * (len(readouts) - 1) / 2
    for i, (label, value) in enumerate(readouts):
        hud.text(x, cy + R + 236, label, 'semi', 22, MUTED, anchor='ms', spacing=5)
        hud.text(x, cy + R + 280, value, 'mono', 34, ICE, anchor='ms')
        if i:
            hud.line([(x - spread / 2, cy + R + 210), (x - spread / 2, cy + R + 284)],
                     CYAN, 1.2, 70, layer='flat')
        x += spread

def draw_standings(hud, data, x0, x1, y0, y1):
    table = data.get('standings') or {}
    drivers = table.get('drivers') or []
    panel(hud, x0, y0, x1, y1, "DRIVERS' CHAMPIONSHIP",
          f"AFTER R{table['round']}" if table.get('round') else '')
    if not drivers:
        hud.text((x0 + x1) / 2, (y0 + y1) / 2, 'NO DATA YET', 'semi', 30, MUTED,
                 anchor='mm', spacing=6)
        return
    lead = max(d['points'] for d in drivers) or 1
    y = y0 + 126
    for d in drivers:
        colour = team_colour(d['team'])
        hud.text(x0 + 30, y, f"{d['position']:02d}", 'mono', 30, MUTED)
        hud.rect(x0 + 84, y - 30, x0 + 90, y + 2, colour, 255, layer='glow')
        hud.text(x0 + 108, y, d['name'].upper(), 'semi', 36, ICE, spacing=2)
        hud.text(x1 - 30, y, points_text(d['points']), 'mono', 34, ICE, anchor='rs')
        gap = d['points'] - lead
        if gap:
            hud.text(x1 - 118, y, points_text(gap), 'mono', 22, MUTED, anchor='rs')
        bar0, bar1 = x0 + 108, x1 - 30
        hud.rect(bar0, y + 14, bar1, y + 17, DIM, 110)
        hud.rect(bar0, y + 14, bar0 + (bar1 - bar0) * d['points'] / lead, y + 17,
                 colour, 235, layer='glow')
        y += 74

    last = data.get('last_race')
    if last:
        hud.line([(x0 + 30, y1 - 84), (x1 - 30, y1 - 84)], CYAN, 1.2, 50, layer='flat')
        hud.text(x0 + 30, y1 - 38, f"R{last['round']}  {short_event(last['name'])}",
                 'semi', 26, MUTED, spacing=3)
        x = x1 - 30
        for place in range(len(last['podium']), 0, -1):
            code = last['podium'][place - 1]
            hud.text(x, y1 - 38, code, 'mono', 28, ICE if place == 1 else CYAN, anchor='rs')
            x -= hud.width_of(code, 'mono', 28) + 8
            hud.text(x, y1 - 38, f"P{place}", 'mono', 20, MUTED, anchor='rs')
            x -= hud.width_of(f"P{place}", 'mono', 20) + 26

def draw_next_race(hud, data, now, x0, x1, y0, y1):
    calendar = data.get('calendar') or {}
    upcoming = calendar.get('next')
    panel(hud, x0, y0, x1, y1, 'NEXT RACE', f"ROUND {upcoming['round']}" if upcoming else '')
    if not upcoming:
        if calendar:
            hud.text((x0 + x1) / 2, (y0 + y1) / 2, 'SEASON COMPLETE', 'display', 36, ICE,
                     anchor='mm', spacing=6)
        else:
            hud.text((x0 + x1) / 2, (y0 + y1) / 2, 'NO DATA YET', 'semi', 30, MUTED,
                     anchor='mm', spacing=6)
        return

    name = upcoming['name'].upper()
    size = 52
    while size > 30 and hud.width_of(name, 'bold', size, 2) > x1 - x0 - 60:
        size -= 2
    hud.text(x0 + 30, y0 + 124, name, 'bold', size, ICE, spacing=2, halo=0.25)
    place = upcoming['location']
    if upcoming['country'] and upcoming['country'].lower() != place.lower():
        place += f"  ·  {upcoming['country']}"
    hud.text(x0 + 30, y0 + 170, place.upper(), 'semi', 28, MUTED, spacing=4)

    start = datetime.fromisoformat(upcoming['start'])
    local = start.astimezone()                           # the laptop's own time zone
    hud.text(x0 + 30, y0 + 268, f"{local:%a %d %b}".upper(), 'display', 42, ICE, spacing=4)
    hud.text(x0 + 30, y0 + 320, f"LIGHTS OUT  {local:%H:%M}", 'semi', 30, CYAN, spacing=4)

    # Countdown gauge: fills up across the gap between the last race and this one
    gx, gy, gr = x1 - 150, y0 + 258, 92
    remaining = start - now
    previous = calendar.get('previous_start')
    gap = start - datetime.fromisoformat(previous) if previous else timedelta(days=14)
    fraction = min(1.0, max(0.0, 1 - remaining / gap)) if gap.total_seconds() > 0 else 1.0
    for i in range(48):
        hud.tick(gx, gy, i * 7.5, gr + 12, gr + 20, CYAN, 1.4, 70)
    hud.ring(gx, gy, gr, DIM, 6, 170)
    hud.ring(gx, gy, gr - 1, RED, 9, 255, 0, 360 * fraction)
    if remaining.total_seconds() <= 0:
        hud.text(gx, gy, 'LIVE', 'display', 36, RED, anchor='mm', halo=0.4)
    elif remaining < timedelta(days=1):
        hours, minutes = divmod(int(remaining.total_seconds() // 60), 60)
        hud.text(gx, gy + 12, f"{hours:02d}:{minutes:02d}", 'mono', 50, ICE, anchor='ms')
        hud.text(gx, gy + 48, 'HRS  MIN', 'semi', 20, MUTED, anchor='ms', spacing=3)
    else:
        days, rest = divmod(int(remaining.total_seconds() // 3600), 24)
        hud.text(gx, gy + 10, f"{days}D", 'mono', 60, ICE, anchor='ms')
        hud.text(gx, gy + 50, f"{rest:02d}H", 'mono', 30, CYAN, anchor='ms')
    hud.text(gx, y1 - 30, 'T-MINUS', 'semi', 22, MUTED, anchor='ms', spacing=6)

def draw_project(hud, data, repo, now, x0, x1, y0, y1):
    panel(hud, x0, y0, x1, y1, 'PROJECT', repo['name'] if repo else '')
    y = y0 + 110
    if repo:
        committed = datetime.fromtimestamp(repo['time'], timezone.utc)
        hud.text(x0 + 30, y, 'LAST COMMIT', 'semi', 24, MUTED, spacing=4)
        hud.text(x1 - 30, y, ago(now - committed), 'mono', 28, ICE, anchor='rs')
        room = x1 - x0 - 60
        if repo.get('uncommitted'):
            note = f"{repo['uncommitted']} UNCOMMITTED"
            hud.text(x1 - 30, y + 48, note, 'mono', 22, AMBER, anchor='rs')
            room -= hud.width_of(note, 'mono', 22) + 30
        hud.text(x0 + 30, y + 48, hud.fit(repo['message'], 'semi', 30, room), 'semi', 30, ICE)
    else:
        hud.text(x0 + 30, y, 'NO GIT HISTORY FOUND', 'semi', 24, MUTED, spacing=4)

    status = (data.get('status') or {}).get('text') or 'ONLINE'
    hud.line([(x0 + 30, y1 - 84), (x1 - 30, y1 - 84)], CYAN, 1.2, 50, layer='flat')
    hud.text(x0 + 30, y1 - 38, 'STATUS', 'semi', 24, MUTED, spacing=4)
    hud.dot(x0 + 150, y1 - 47, 5, CYAN)
    hud.text(x0 + 170, y1 - 38, hud.fit(status.upper(), 'semi', 28, x1 - x0 - 200, 2),
             'semi', 28, CYAN, spacing=2)

def render(data, width=BASE_W, height=BASE_H, now=None, repo=None):
    now = now or datetime.now(timezone.utc)
    hud = Hud(width, height)
    left, right = 400, hud.w - 90                        # left 400 kept clear for icons
    panel_x = right - 780
    cx, cy, R = (left + panel_x - 40) / 2, 820, 500

    draw_backdrop(hud, left, right, cx, cy, R)
    draw_header(hud, data, now, left, right)
    draw_ring(hud, data, cx, cy, R, left, panel_x)
    draw_track(hud, data.get('track'), cx, cy, R)
    draw_standings(hud, data, panel_x, right, 200, 830)
    draw_next_race(hud, data, now, panel_x, right, 880, 1300)
    draw_project(hud, data, repo, now, panel_x, right, 1350, 1640)

    return hud.finish(background(width, height, (cx * hud.k, (cy + hud.dy) * hud.k)))

# --- Windows -----------------------------------------------------------------

def screen_size():
    """Real pixel size of the main screen (not the 200%-scaled size Windows reports)."""
    if sys.platform != 'win32':
        return BASE_W, BASE_H
    import ctypes
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    DESKTOPVERTRES, DESKTOPHORZRES = 117, 118
    hdc = user32.GetDC(0)
    try:
        width, height = gdi32.GetDeviceCaps(hdc, DESKTOPHORZRES), gdi32.GetDeviceCaps(hdc, DESKTOPVERTRES)
    finally:
        user32.ReleaseDC(0, hdc)
    return (width, height) if width and height else (BASE_W, BASE_H)

def set_wallpaper(path):
    if sys.platform != 'win32':
        log('not on Windows, so the image was saved but not set as the wallpaper')
        return
    import ctypes
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Control Panel\Desktop', 0,
                        winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, 'WallpaperStyle', 0, winreg.REG_SZ, '10')   # fill
        winreg.SetValueEx(key, 'TileWallpaper', 0, winreg.REG_SZ, '0')
    SPI_SETDESKWALLPAPER, SAVE_AND_BROADCAST = 20, 0x01 | 0x02
    if not ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, str(path),
                                                      SAVE_AND_BROADCAST):
        log('Windows refused to set the wallpaper')

def install():
    """Task Scheduler job that redraws every REFRESH_MINUTES, on battery too."""
    from lockscreen import register_task       # shared helper, lives next to this file
    register_task(TASK_NAME, __file__, REFRESH_MINUTES)
    log(f"installed '{TASK_NAME}', redrawing every {REFRESH_MINUTES} minutes")

def uninstall():
    from lockscreen import unregister_task
    unregister_task(TASK_NAME)
    log(f"removed '{TASK_NAME}'")

# --- Entry points ------------------------------------------------------------

_lock = threading.Lock()

def update(status=None, fetch=None, set_background=True):
    """Redraw the wallpaper. Passing just a status redraws from saved data,
    so it takes a couple of seconds instead of waiting on the internet."""
    with _lock:
        if fetch is None:
            fetch = status is None
        saved = load_saved()
        data = saved
        if fetch:
            try:
                data = fetch_all(saved)
            except Exception as error:     # e.g. FastF1 missing or broken
                log(f"fetch failed ({error}), drawing from saved data")
        if status is not None:
            data['status'] = {'text': status, 'time': datetime.now(timezone.utc).isoformat()}
        save(data)
        image = render(data, *screen_size(), repo=repo_info())
        temporary = OUTPUT.with_name('hud_wallpaper.tmp.png')
        image.save(temporary)
        temporary.replace(OUTPUT)
        if set_background:
            set_wallpaper(OUTPUT)
        return OUTPUT

def update_in_background(status=None):
    """Same as update(), on a background thread so the caller never waits."""
    thread = threading.Thread(target=update, kwargs={'status': status}, daemon=True)
    thread.start()
    return thread

def main():
    parser = argparse.ArgumentParser(description='F1 desktop HUD wallpaper')
    parser.add_argument('--no-set', action='store_true', help="draw it but don't set it")
    parser.add_argument('--offline', action='store_true', help='reuse saved data')
    parser.add_argument('--status', help='text for the STATUS line')
    parser.add_argument('--install', action='store_true', help='redraw automatically')
    parser.add_argument('--uninstall', action='store_true', help='stop redrawing')
    args = parser.parse_args()

    if args.install:
        install()
    elif args.uninstall:
        uninstall()
    else:
        path = update(status=args.status, fetch=not args.offline,
                      set_background=not args.no_set)
        log(f'wallpaper saved to {path}')

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log(f'failed: {error!r}')
        raise
