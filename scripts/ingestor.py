# scripts/ingestor.py
import os
import gc
import shutil
from datetime import datetime
import fastf1
import pandas as pd

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def run_ingestion(start_year=2018, end_year=None, force_refresh=False):
    if end_year is None:
        end_year = datetime.now().year

    raw_path = 'data/raw/pro_f1_raw.parquet'
    existing_races = set()
    existing_df = None

    if os.path.exists(raw_path) and not force_refresh:
        try:
            existing_df = pd.read_parquet(raw_path)
            if not existing_df.empty:
                existing_races = set(zip(existing_df['Season'].astype(int), existing_df['Round'].astype(int)))
        except Exception as read_err:
            print(f"Notice: Could not read existing dataset ({read_err}). Full refresh will be performed.")

    # Determine which years need schedule inspection
    current_year = datetime.now().year
    years_in_existing = {yr for yr, _ in existing_races}

    if not force_refresh and existing_races:
        # Check only past years that have zero ingested races or the active current year
        years_to_check = [
            yr for yr in range(start_year, end_year + 1)
            if yr not in years_in_existing or yr == current_year
        ]
    else:
        years_to_check = list(range(start_year, end_year + 1))

    temp_dir = 'data/raw/temp_races'
    os.makedirs(temp_dir, exist_ok=True)
    
    new_races_count = 0
    today = datetime.now().date()
    
    for year in years_to_check:
        try:
            schedule = fastf1.get_event_schedule(year)
            race_events = schedule[schedule['EventFormat'] != 'testing']
        except Exception as e:
            print(f"Error loading schedule for {year}: {e}")
            continue
        
        for _, event in race_events.iterrows():
            round_num = int(event['RoundNumber'])
            
            # Fast skip if already ingested
            if not force_refresh and (year, round_num) in existing_races:
                continue
                
            # Skip future events whose EventDate has not passed yet
            if 'EventDate' in event and pd.notnull(event['EventDate']):
                event_date = pd.to_datetime(event['EventDate']).date()
                if event_date > today:
                    continue
                
            temp_file_path = os.path.join(temp_dir, f"{year}_R{round_num}.parquet")
            if os.path.exists(temp_file_path) and not force_refresh:
                print(f"Already processed: {year} R{round_num} - {event['EventName']}")
                continue
                
            session_data = []
            session = None
            try:
                print(f"Loading new race: {year} R{round_num} - {event['EventName']}")
                session = fastf1.get_session(year, round_num, 'R')
                try:
                    session.load(laps=True, telemetry=False, weather=True)
                except Exception as load_err:
                    print(f"Notice: Laps not available for {year} R{round_num} ({load_err}). Retrying with laps=False...")
                    session.load(laps=False, telemetry=False, weather=True)
                
                results = session.results
                if results is None or results.empty:
                    session.load(laps=False, telemetry=False, weather=False)
                    results = session.results
                
                if results is None or results.empty:
                    print(f"No results found for {year} R{round_num} ({event['EventName']}). Skipping.")
                    continue

                weather = session.weather_data
                is_sprint = 1 if 'Sprint' in str(event.get('EventFormat', '')) else 0
                
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
                        'EventName': event.get('EventName', ''),
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
                    new_races_count += 1
            except Exception as e:
                print(f"Error loading {year} R{round_num}: {e}")
            finally:
                # Discard session object and clean memory
                del session
                del session_data
                gc.collect()

    all_temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.parquet')]

    # Fast Exit if raw dataset is already complete and no new race temp files were created
    if not all_temp_files and existing_df is not None and not existing_df.empty and not force_refresh:
        total_races = len(existing_df.groupby(['Season', 'Round']))
        max_s = existing_df['Season'].max()
        max_r = existing_df[existing_df['Season'] == max_s]['Round'].max()
        print(f"[INFO] Raw dataset is already up to date ({total_races} races up to {max_s} R{max_r}). Skipping download.")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {
            'new_races_ingested': 0,
            'total_races': total_races,
            'raw_path': raw_path,
            'up_to_date': True
        }

    # Consolidate temporary race files into raw dataset
    print("Consolidating race files into data/raw/pro_f1_raw.parquet...")
    dfs = []
    if os.path.exists(raw_path):
        dfs.append(pd.read_parquet(raw_path))
        
    for file in all_temp_files:
        dfs.append(pd.read_parquet(file))
        
    total_races = 0
    if dfs:
        consolidated_df = pd.concat(dfs, ignore_index=True)
        consolidated_df = consolidated_df.drop_duplicates(subset=['Season', 'Round', 'Driver'], keep='last')
        consolidated_df = consolidated_df.sort_values(['Season', 'Round']).reset_index(drop=True)
        
        os.makedirs('data/raw', exist_ok=True)
        consolidated_df.to_parquet(raw_path, index=False)
        total_races = len(consolidated_df.groupby(['Season', 'Round']))
        print(f"Successfully saved consolidated dataset ({len(consolidated_df)} rows across {total_races} races) to {raw_path}")
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory data/raw/temp_races")
    else:
        print("No race files found to consolidate.")
        
    return {
        'new_races_ingested': new_races_count,
        'total_races': total_races,
        'raw_path': raw_path,
        'up_to_date': False
    }

def get_upcoming_race_grid(year=None, round_num=None):
    """
    Fetches starting grid data for an upcoming race round (from Qualifying).
    Returns a pandas DataFrame formatted identically to raw race rows.
    """
    if year is None:
        year = datetime.now().year

    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[schedule['EventFormat'] != 'testing']

    target_event = None
    if round_num is not None:
        matched = race_events[race_events['RoundNumber'] == round_num]
        if not matched.empty:
            target_event = matched.iloc[0]
    else:
        # Find the first event that hasn't completed or next upcoming round
        raw_path = 'data/raw/pro_f1_raw.parquet'
        existing_rounds = set()
        if os.path.exists(raw_path):
            existing_df = pd.read_parquet(raw_path)
            if not existing_df.empty:
                existing_rounds = set(existing_df[existing_df['Season'] == year]['Round'].unique())
        
        for _, evt in race_events.iterrows():
            r_num = int(evt['RoundNumber'])
            if r_num not in existing_rounds:
                target_event = evt
                round_num = r_num
                break

    if target_event is None:
        print(f"[INGESTOR] No upcoming or uningested race found for {year}.")
        return None

    print(f"[INGESTOR] Fetching pre-race grid for {year} Round {round_num} - {target_event['EventName']}...")
    
    session_data = []
    try:
        # Try loading Qualifying session
        session_q = fastf1.get_session(year, round_num, 'Q')
        try:
            session_q.load(laps=False, telemetry=False, weather=True)
        except Exception:
            session_q.load(laps=False, telemetry=False, weather=False)

        q_results = session_q.results
        if q_results is None or q_results.empty:
            print(f"[INGESTOR WARNING] Qualifying results not available yet for {year} R{round_num}.")
            return None

        weather = session_q.weather_data
        is_sprint = 1 if 'Sprint' in str(target_event.get('EventFormat', '')) else 0
        mean_track_temp = 25.0
        has_rain = 0

        if weather is not None and not weather.empty:
            if 'TrackTemp' in weather.columns:
                mean_track_temp = weather['TrackTemp'].mean()
            if 'Rainfall' in weather.columns:
                has_rain = 1 if weather['Rainfall'].any() else 0

        for _, driver in q_results.iterrows():
            pos = driver.get('Position', driver.get('ClassifiedPosition', 20))
            try:
                grid_pos = int(pos)
            except (ValueError, TypeError):
                grid_pos = 20

            session_data.append({
                'Season': year,
                'Round': round_num,
                'Circuit': target_event['Location'],
                'EventName': target_event.get('EventName', ''),
                'Driver': driver['Abbreviation'],
                'Team': driver['TeamName'],
                'GridPosition': grid_pos,
                'FinishPos': grid_pos,  # Expected dummy position for rolling metrics
                'Status': 'Finished',
                'LapsCompleted': 0,
                'TrackTemp': mean_track_temp,
                'Rain': has_rain,
                'IsSprint': is_sprint
            })

    except Exception as err:
        print(f"[INGESTOR ERROR] Failed to fetch starting grid for {year} R{round_num}: {err}")
        return None

    if session_data:
        df_grid = pd.DataFrame(session_data)
        print(f"[INGESTOR] Successfully fetched {len(df_grid)} driver starting slots for {year} R{round_num}.")
        return df_grid

    return None

if __name__ == "__main__":
    run_ingestion()