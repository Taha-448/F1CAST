# scripts/ingestor.py
import os
import gc
import shutil
import fastf1
import pandas as pd

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def run_ingestion(start_year, end_year):
    temp_dir = 'data/raw/temp_races'
    os.makedirs(temp_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
            race_events = schedule[schedule['EventFormat'] != 'testing']
        except Exception as e:
            print(f"Error loading schedule for {year}: {e}")
            continue
        
        for _, event in race_events.iterrows():
            round_num = event['RoundNumber']
            
            # Skip rounds > 9 for 2026 as per requirements
            if year == 2026 and round_num > 9:
                continue
                
            temp_file_path = os.path.join(temp_dir, f"{year}_R{round_num}.parquet")
            if os.path.exists(temp_file_path):
                print(f"Already processed: {year} R{round_num} - {event['EventName']}")
                continue
                
            session_data = []
            session = None
            try:
                print(f"Loading: {year} R{round_num} - {event['EventName']}")
                session = fastf1.get_session(year, round_num, 'R')
                session.load(laps=True, telemetry=False, weather=True)
                
                results = session.results
                weather = session.weather_data
                is_sprint = 1 if 'Sprint' in event['EventFormat'] else 0
                
                # Fetch mean TrackTemp and Rain availability safely
                mean_track_temp = 25.0
                has_rain = 0
                if weather is not None and not weather.empty:
                    if 'TrackTemp' in weather.columns:
                        mean_track_temp = weather['TrackTemp'].mean()
                    if 'Rainfall' in weather.columns:
                        has_rain = 1 if weather['Rainfall'].any() else 0

                for _, driver in results.iterrows():
                    session_data.append({
                        'Season': year,
                        'Round': round_num,
                        'Circuit': event['Location'],
                        'Driver': driver['Abbreviation'],
                        'Team': driver['TeamName'],
                        'GridPosition': driver['GridPosition'],
                        'FinishPos': driver['ClassifiedPosition'],
                        'Status': driver['Status'],
                        'LapsCompleted': driver['Laps'],
                        'TrackTemp': mean_track_temp,
                        'Rain': has_rain,
                        'IsSprint': is_sprint
                    })
                
                # Write individual race data to temp Parquet file
                if session_data:
                    pd.DataFrame(session_data).to_parquet(temp_file_path, index=False)
                    print(f"Saved: {temp_file_path}")
            except Exception as e:
                print(f"Error loading {year} R{round_num}: {e}")
            finally:
                # Discard session object and clean memory
                del session
                del session_data
                gc.collect()
                
    # Consolidate all parquet files into a single parquet file
    print("Consolidating all temporary race files...")
    all_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.parquet')]
    if all_files:
        dfs = []
        for file in all_files:
            dfs.append(pd.read_parquet(file))
        consolidated_df = pd.concat(dfs, ignore_index=True)
        
        # Save to final parquet location
        os.makedirs('data/raw', exist_ok=True)
        consolidated_df.to_parquet('data/raw/pro_f1_raw.parquet', index=False)
        print("Successfully saved consolidated dataset to data/raw/pro_f1_raw.parquet")
        
        # Clean up temp files and directory
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary directory data/raw/temp_races")
    else:
        print("No race files found to consolidate.")

if __name__ == "__main__":
    run_ingestion(2018, 2026)