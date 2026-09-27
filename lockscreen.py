"""Lock screen slideshow for the F1 Analysis Assistant.

Windows' own lock screen slideshow won't read pictures kept in OneDrive, so
this does the job itself: every INTERVAL_MINUTES it sets the next picture from
PICTURES as the lock screen, using Windows' lock screen API through PowerShell.
Standard library only, so it's cheap enough to run every minute.

Usage:
    python lockscreen.py              show the next picture now
    python lockscreen.py --install    switch pictures every INTERVAL_MINUTES
    python lockscreen.py --uninstall  stop switching

Set Settings > Personalisation > Lock screen to 'Picture' first.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PICTURES = Path.home() / 'Documents' / 'F1 Lock Screens'   # any .jpg/.png in here
INTERVAL_MINUTES = 1
TASK_NAME = 'F1 Lock Screen'
IMAGE_TYPES = ('.jpg', '.jpeg', '.png')

HERE = Path(__file__).resolve().parent
STATE_DIR = (Path(os.environ['LOCALAPPDATA']) / 'F1DesktopHUD'
             if os.environ.get('LOCALAPPDATA') else HERE)
STATE_FILE = STATE_DIR / 'lockscreen.json'
LOG_FILE = STATE_DIR / 'lockscreen.log'

# --- PowerShell --------------------------------------------------------------

def powershell(script, **env):
    """Run a Windows PowerShell script with no console window. Values go in
    through environment variables, so paths with spaces or Arabic letters are safe."""
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
         '-EncodedCommand', encoded],
        env={**os.environ, **env}, capture_output=True, text=True, errors='replace',
        timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return result.returncode == 0, (result.stderr or result.stdout).strip()

SET_LOCKSCREEN = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$methods = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 }
$asTaskOperation = $methods | Where-Object {
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
$asTaskAction = $methods | Where-Object {
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' } | Select-Object -First 1
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.System.UserProfile.LockScreen, Windows.System.UserProfile, ContentType = WindowsRuntime] | Out-Null
$open = [Windows.Storage.StorageFile]::GetFileFromPathAsync($env:LOCK_IMAGE)
$task = $asTaskOperation.MakeGenericMethod([Windows.Storage.StorageFile]).Invoke($null, @($open))
$task.Wait(-1) | Out-Null
$set = [Windows.System.UserProfile.LockScreen]::SetImageFileAsync($task.Result)
$asTaskAction.Invoke($null, @($set)).Wait(-1) | Out-Null
"""

# Task Scheduler job that also runs on battery and catches up after sleep.
REGISTER_TASK = r"""
$ErrorActionPreference = 'Stop'
$action = New-ScheduledTaskAction -Execute $env:TASK_EXE -Argument $env:TASK_ARGS `
    -WorkingDirectory $env:TASK_DIR
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes ([int]$env:TASK_MINUTES))
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $env:TASK_NAME -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
"""

UNREGISTER_TASK = r"""
$ErrorActionPreference = 'Stop'
Unregister-ScheduledTask -TaskName $env:TASK_NAME -Confirm:$false
"""

def register_task(name, script, minutes):
    """Run `script` with pythonw (no window) every `minutes`, on battery too."""
    pythonw = Path(sys.executable).with_name('pythonw.exe')
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    ok, output = powershell(REGISTER_TASK, TASK_NAME=name, TASK_EXE=str(exe),
                            TASK_ARGS=f'"{Path(script).resolve()}"',
                            TASK_DIR=str(Path(script).resolve().parent),
                            TASK_MINUTES=str(minutes))
    if not ok:
        raise RuntimeError(f"couldn't create the scheduled task: {output}")

def unregister_task(name):
    ok, output = powershell(UNREGISTER_TASK, TASK_NAME=name)
    if not ok:
        raise RuntimeError(f"couldn't remove the scheduled task: {output}")

# --- Slideshow ---------------------------------------------------------------

def log(message):
    """Keep the last 100 lines. Only problems and installs are logged, not every change."""
    line = f"[{datetime.now():%d %b %H:%M}] {message}"
    print(line)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        old = LOG_FILE.read_text(encoding='utf-8').splitlines()[-99:] if LOG_FILE.exists() else []
        LOG_FILE.write_text('\n'.join(old + [line]) + '\n', encoding='utf-8')
    except OSError:
        pass

def show_next():
    """Set the next picture in PICTURES as the lock screen. Returns its name."""
    pictures = sorted(p for p in PICTURES.glob('*') if p.suffix.lower() in IMAGE_TYPES)
    if not pictures:
        log(f"no pictures found in {PICTURES}")
        return None
    try:
        index = json.loads(STATE_FILE.read_text(encoding='utf-8'))['index']
    except (OSError, ValueError, KeyError):
        index = -1
    index = (index + 1) % len(pictures)
    ok, output = powershell(SET_LOCKSCREEN, LOCK_IMAGE=str(pictures[index]))
    if not ok:
        log(f"couldn't set {pictures[index].name}: {output[:400]}")
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({'index': index}), encoding='utf-8')
    return pictures[index].name

def main():
    parser = argparse.ArgumentParser(description='F1 lock screen slideshow')
    parser.add_argument('--install', action='store_true',
                        help=f'switch pictures every {INTERVAL_MINUTES} minute(s)')
    parser.add_argument('--uninstall', action='store_true', help='stop switching')
    args = parser.parse_args()

    if sys.platform != 'win32':
        sys.exit('This only works on Windows.')
    if args.install:
        register_task(TASK_NAME, __file__, INTERVAL_MINUTES)
        log(f"installed '{TASK_NAME}': next picture every {INTERVAL_MINUTES} minute(s)")
    elif args.uninstall:
        unregister_task(TASK_NAME)
        log(f"removed '{TASK_NAME}'")
    else:
        name = show_next()
        if name:
            print(f"lock screen set to {name}")

if __name__ == '__main__':
    main()
