"""Record and replay F1 live timing.

Recording requires an active F1TV Access/Pro/Premium subscription. The client
will print an authentication URL on first run.

Usage:
    python live.py                      # records to live_data.txt
    python live.py monza_2026.txt       # records to a named file

Afterwards, load the recording with load_recording() and every analysis and
visual function in this project works on it unchanged.
"""

import sys

import fastf1
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

def record(output_file='live_data.txt'):
    """Connect to the live timing stream and write it to disk.

    Start 10-15 minutes before the session begins so the setup data is
    captured, and leave it running until the session ends. Closing the
    terminal or letting the machine sleep will truncate the recording.
    """
    print(f"Recording live timing to {output_file}")
    print("Start before the session begins. Press Ctrl+C to stop.\n")

    client = SignalRClient(filename=output_file)
    client.start()

def load_recording(year, race, session_type='R', source='live_data.txt'):
    """Load a session from a recorded live timing file.

    Returns a session object identical in shape to a normal FastF1 session,
    so analyse_stints, corner_analysis, the speed maps and everything else
    work on it without modification.
    """
    livedata = LiveTimingData(source)
    session = fastf1.get_session(year, race, session_type)
    session.load(livedata=livedata)
    return session

if __name__ == '__main__':
    output = sys.argv[1] if len(sys.argv) > 1 else 'live_data.txt'
    record(output)
    