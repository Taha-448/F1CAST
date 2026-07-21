# scripts/state.py
"""
Pipeline State Manager for F1CAST

Manages `state.json` at project root to track pipeline execution status,
prediction dispatch states, and comparison email notifications.
"""

import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE_PATH = os.path.join(PROJECT_ROOT, 'state.json')

DEFAULT_STATE = {
    "last_processed_season": None,
    "last_processed_round": None,
    "predictions_sent": {},
    "comparisons_sent": {},
    "last_updated": None
}

def load_state(state_file=STATE_FILE_PATH):
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                # Ensure all required keys exist
                for k, v in DEFAULT_STATE.items():
                    if k not in state:
                        state[k] = v
                return state
        except Exception as e:
            print(f"[STATE WARNING] Could not read state file ({e}). Initializing default state.")
    return DEFAULT_STATE.copy()

def save_state(state, state_file=STATE_FILE_PATH):
    state["last_updated"] = datetime.now().isoformat()
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        print(f"[STATE] Successfully saved pipeline state to {os.path.basename(state_file)}")
    except Exception as e:
        print(f"[STATE ERROR] Failed to save state file ({e})")

def get_race_key(season, round_num):
    return f"{season}_R{round_num}"

def is_prediction_sent(season, round_num, state_file=STATE_FILE_PATH):
    state = load_state(state_file)
    key = get_race_key(season, round_num)
    return state.get("predictions_sent", {}).get(key, False)

def mark_prediction_sent(season, round_num, state_file=STATE_FILE_PATH):
    state = load_state(state_file)
    key = get_race_key(season, round_num)
    if "predictions_sent" not in state:
        state["predictions_sent"] = {}
    state["predictions_sent"][key] = True
    state["last_processed_season"] = season
    state["last_processed_round"] = round_num
    save_state(state, state_file)

def is_comparison_sent(season, round_num, state_file=STATE_FILE_PATH):
    state = load_state(state_file)
    key = get_race_key(season, round_num)
    return state.get("comparisons_sent", {}).get(key, False)

def mark_comparison_sent(season, round_num, state_file=STATE_FILE_PATH):
    state = load_state(state_file)
    key = get_race_key(season, round_num)
    if "comparisons_sent" not in state:
        state["comparisons_sent"] = {}
    state["comparisons_sent"][key] = True
    state["last_processed_season"] = season
    state["last_processed_round"] = round_num
    save_state(state, state_file)

if __name__ == "__main__":
    st = load_state()
    print("Current State:", json.dumps(st, indent=2))
