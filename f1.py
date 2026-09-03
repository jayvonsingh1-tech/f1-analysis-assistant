import fastf1
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2024, 'Monza', 'R')
session.load()

nor = session.laps.pick_drivers('NOR')
pia = session.laps.pick_drivers('PIA')

plt.plot(nor['LapNumber'], nor['LapTime'].dt.total_seconds(), label='Norris')
plt.plot(pia['LapNumber'], pia['LapTime'].dt.total_seconds(), label='Piastri')

plt.xlabel('Lap')
plt.ylabel('Lap time (s)')
plt.title('Norris vs Piastri - Monza 2024')
plt.legend()
plt.show()
