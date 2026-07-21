import os
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt

# 1. Setup caching (highly recommended, as F1 data files can be 50-100MB)
# This avoids re-downloading the data every time you run the script.
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2. Setup fastf1's built-in matplotlib formatting for F1 colors/dark theme
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

print("Loading 2026 British Grand Prix data...")
# Load the 2026 Silverstone Session ('R' stands for Race)
session = fastf1.get_session(2026, 'Spa', 'R')
session.load()

# --- Part 1: Display Race Results ---
print("\n=== FINAL RACE RESULTS ===")
results = session.results

# Select specific columns to display cleanly
summary_results = results[['Position', 'Abbreviation', 'TeamName', 'GridPosition', 'Status', 'Points']]
print(summary_results.to_string(index=False))

# --- Part 2: Plot Lap Times for the Top 3 Finishers ---
print("\nGenerating lap time plot for the podium finishers...")

# Get all laps from the session
laps = session.laps

# Identify the top 3 drivers from the results
top_3_drivers = results['Abbreviation'].head(3).tolist()

fig, ax = plt.subplots(figsize=(10, 6))

for driver in top_3_drivers:
    # Filter laps for the specific driver and pick quick laps (excluding pit-out/pit-in anomalies)
    driver_laps = laps.pick_driver(driver).pick_quicklaps()
    
    # Get the official team color
    team_color = fastf1.plotting.get_team_color(driver_laps['Team'].iloc[0], session=session)
    
    # Plot Lap Number vs Lap Time (converted to seconds)
    ax.plot(
        driver_laps['LapNumber'], 
        driver_laps['LapTime'].dt.total_seconds(), 
        label=driver, 
        color=team_color,
        linewidth=2
    )

# Formatting the plot
ax.set_title(f"{session.event['EventName']} {session.event.year} - Top 3 Lap Time Progression")
ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time (seconds)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()
