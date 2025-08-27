
#!/usr/bin/env python3
"""
NFL 10-Year Game Dataset Builder
--------------------------------
Downloads schedule + betting lines from nflverse/nfldata and computes
home/away pregame records for each matchup.

Output: nfl_games_10yr_records_spreads.csv
Default seasons: last 10 completed.

Columns (selected):
- season, week, game_type, weekday, gameday, gametime, game_time_12hr, game_id
- home_team, away_team, location
- spread_line, away_spread_odds, home_spread_odds, total_line, over_odds, under_odds
- home_record_pre_W, home_record_pre_L, home_record_pre_T, home_record_pre_pct
- away_record_pre_W, away_record_pre_L, away_record_pre_T, away_record_pre_pct
- home_score, away_score, away_moneyline, home_moneyline
- roof, surface, temp, away_rest, home_rest

Usage:
    python nfl_10yr_pipeline.py                # builds last 10 seasons
    python nfl_10yr_pipeline.py --start 2015 --end 2024

Requires:
    pip install pandas pyarrow requests
"""

import argparse
import io
import sys
from datetime import datetime, date
from typing import List

import pandas as pd
import requests


RAW_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _convert_to_12hr(time_str: str) -> str:
    """Convert 24-hour time format (HH:MM) to 12-hour format (H:MM AM/PM)."""
    try:
        if pd.isna(time_str) or time_str == '':
            return time_str
        
        # Parse the time string
        time_obj = pd.to_datetime(time_str, format='%H:%M').time()
        
        # Convert to 12-hour format
        if time_obj.hour == 0:
            return f"12:{time_obj.minute:02d} AM"
        elif time_obj.hour < 12:
            return f"{time_obj.hour}:{time_obj.minute:02d} AM"
        elif time_obj.hour == 12:
            return f"12:{time_obj.minute:02d} PM"
        else:
            return f"{time_obj.hour - 12}:{time_obj.minute:02d} PM"
    except:
        return time_str


def _season_range(start: int, end: int) -> List[int]:
    if start > end:
        raise ValueError("start season must be <= end season")
    return list(range(start, end + 1))


def _download_games_csv() -> pd.DataFrame:
    resp = requests.get(RAW_GAMES_URL, timeout=60)
    resp.raise_for_status()
    # Read directly into pandas
    return pd.read_csv(io.BytesIO(resp.content))


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Many nfldata columns are already present; we ensure gameday is a timestamp and week is int
    df = df.copy()
    # gameday is 'YYYY-MM-DD' or may be missing for some records (future games). Coerce errors to NaT.
    df['gameday'] = pd.to_datetime(df.get('gameday'), errors='coerce')
    
    # Process gametime if available
    if 'gametime' in df.columns:
        # Convert 24-hour time to 12-hour format for readability
        df['game_time_12hr'] = df['gametime'].apply(lambda x: _convert_to_12hr(x) if pd.notna(x) else x)
        
        # Combine into full datetime if available for sorting
        # If gametime missing, keep date only. Timezone not needed for ordering within season.
        time_parts = pd.to_datetime(df['gametime'], format='%H:%M', errors='coerce').dt.time
        df['game_dt'] = df['gameday']
        try:
            df.loc[time_parts.notna(), 'game_dt'] = pd.to_datetime(
                df['gameday'].dt.strftime('%Y-%m-%d') + ' ' + df['gametime'],
                errors='coerce'
            )
        except Exception:
            df['game_dt'] = df['gameday']
    else:
        df['game_dt'] = df['gameday']
        df['game_time_12hr'] = pd.NA
    
    return df


def _compute_team_results_long(df_games: pd.DataFrame) -> pd.DataFrame:
    """Return a long dataframe with one row per team per game with the result from that team's perspective."""
    g = df_games.copy()

    # Margin (home perspective): result column in nfldata is home_score - away_score
    # Fall back to computing it if missing
    if 'result' in g.columns and g['result'].notna().any():
        margin_home = g['result']
    else:
        margin_home = g['home_score'].fillna(0) - g['away_score'].fillna(0)

    # Build home-side perspective
    home = pd.DataFrame({
        'game_id': g['game_id'],
        'season': g['season'],
        'game_type': g['game_type'],
        'week': g['week'],
        'game_dt': g['game_dt'],
        'team': g['home_team'],
        'opponent': g['away_team'],
        'is_home': True,
        'is_away': False,
        'is_neutral_site': (g['location'].str.lower() == 'neutral') if 'location' in g.columns else False,
        'margin': margin_home
    })

    # Away perspective
    away = pd.DataFrame({
        'game_id': g['game_id'],
        'season': g['season'],
        'game_type': g['game_type'],
        'week': g['week'],
        'game_dt': g['game_dt'],
        'team': g['away_team'],
        'opponent': g['home_team'],
        'is_home': False,
        'is_away': True,
        'is_neutral_site': (g['location'].str.lower() == 'neutral') if 'location' in g.columns else False,
        'margin': -margin_home
    })

    long_df = pd.concat([home, away], ignore_index=True)

    # Win/Loss/Tie flags from the team's perspective
    long_df['W'] = (long_df['margin'] > 0).astype(int)
    long_df['L'] = (long_df['margin'] < 0).astype(int)
    long_df['T'] = (long_df['margin'] == 0).astype(int)

    return long_df


def _compute_pregame_records(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative W/L/T *before* the given game."""
    df = long_df.sort_values(['season', 'team', 'game_dt', 'week', 'game_id']).copy()
    
    # Initialize cumulative columns
    df['cum_W'] = 0
    df['cum_L'] = 0
    df['cum_T'] = 0
    df['cum_pct'] = pd.NA
    
    # Group by season and team to compute cumulative totals
    for (season, team), group in df.groupby(['season', 'team']):
        # Get indices for this group
        idx = group.index
        
        # Compute cumulative W/L/T shifted by 1 (pregame)
        cum_W = group['W'].cumsum().shift(fill_value=0)
        cum_L = group['L'].cumsum().shift(fill_value=0)
        cum_T = group['T'].cumsum().shift(fill_value=0)
        
        # Assign back to the main dataframe
        df.loc[idx, 'cum_W'] = cum_W
        df.loc[idx, 'cum_L'] = cum_L
        df.loc[idx, 'cum_T'] = cum_T
        
        # Compute percentage avoiding divide-by-zero
        denom = cum_W + cum_L + cum_T
        mask = denom > 0
        df.loc[idx[mask], 'cum_pct'] = cum_W[mask] / denom[mask]
    
    return df


def build_dataset(start_season: int, end_season: int, out_csv: str) -> None:
    seasons = _season_range(start_season, end_season)

    games = _download_games_csv()
    games = _normalize_dates(games)

    # Filter seasons and keep only completed games (scores not null)
    games = games[games['season'].isin(seasons)].copy()
    if 'home_score' in games.columns and 'away_score' in games.columns:
        games = games[games['home_score'].notna() & games['away_score'].notna()].copy()

    # Keep useful columns
    keep_cols = [
        'game_id','season','game_type','week','gameday','game_dt','gametime','game_time_12hr',
        'weekday','home_team','home_score','away_team','away_score','location',
        'spread_line','away_spread_odds','home_spread_odds','total_line','over_odds','under_odds',
        'away_moneyline','home_moneyline','result','roof','surface','temp','away_rest','home_rest'
    ]
    keep_cols = [c for c in keep_cols if c in games.columns]
    games = games[keep_cols].copy()

    # Compute long-form team results and pregame records
    long_df = _compute_team_results_long(games)
    pre_df = _compute_pregame_records(long_df)

    # Merge back to game-level for home/away records
    # Home pregame
    home_pre = pre_df[pre_df['is_home']][['game_id','team','cum_W','cum_L','cum_T','cum_pct']].copy()
    home_pre = home_pre.rename(columns={
        'team':'home_team',
        'cum_W':'home_record_pre_W',
        'cum_L':'home_record_pre_L',
        'cum_T':'home_record_pre_T',
        'cum_pct':'home_record_pre_pct'
    })

    # Away pregame
    away_pre = pre_df[pre_df['is_away']][['game_id','team','cum_W','cum_L','cum_T','cum_pct']].copy()
    away_pre = away_pre.rename(columns={
        'team':'away_team',
        'cum_W':'away_record_pre_W',
        'cum_L':'away_record_pre_L',
        'cum_T':'away_record_pre_T',
        'cum_pct':'away_record_pre_pct'
    })

    out = games.merge(home_pre, on=['game_id','home_team'], how='left') \
               .merge(away_pre, on=['game_id','away_team'], how='left')

    # Friendly ordering / selection
    select = [
        'season','game_type','week','weekday','gameday','gametime','game_time_12hr','game_id',
        'home_team','away_team','location','spread_line','away_spread_odds','home_spread_odds',
        'home_record_pre_W','home_record_pre_L','home_record_pre_T','home_record_pre_pct',
        'away_record_pre_W','away_record_pre_L','away_record_pre_T','away_record_pre_pct',
        'home_score','away_score','total_line','over_odds','under_odds','away_moneyline','home_moneyline',
        'roof','surface','temp','away_rest','home_rest'
    ]
    select = [c for c in select if c in out.columns]
    out = out[select].sort_values(['season','game_type','week','gameday','game_id'])

    # Save
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} rows to {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build 10-year NFL game dataset with spreads and pregame team records.")
    default_end = date.today().year - 1
    default_start = default_end - 9
    parser.add_argument('--start', type=int, default=default_start, help='First season (e.g., 2015)')
    parser.add_argument('--end', type=int, default=default_end, help='Last season (e.g., 2024)')
    parser.add_argument('--out', type=str, default='nfl_games_10yr_records_spreads.csv', help='Output CSV filename')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.start, args.end, args.out)
