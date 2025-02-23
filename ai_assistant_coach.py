import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from statsbombpy import sb

# Fetch StatsBomb Data (2022 WC Final)
events = sb.events(match_id=3869685)
shots_df = events[events['type'] == 'Shot'].copy()
shots_df['distance'] = np.sqrt((120 - shots_df['location'].apply(lambda x: x[0])) ** 2 +
                               (40 - shots_df['location'].apply(lambda x: x[1])) ** 2)
shots_df['angle'] = np.arctan2(40 - shots_df['location'].apply(lambda x: x[1]),
                               120 - shots_df['location'].apply(lambda x: x[0])).abs() * 180 / np.pi
shots_df['goal'] = shots_df['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

# Train xG Model
X = shots_df[['distance', 'angle']]
y = shots_df['goal']
model = LogisticRegression().fit(X, y)
shots_df['xG'] = model.predict_proba(X)[:, 1]

# Team Stats
teams_df = pd.DataFrame({
    'Team': ['Argentina', 'France'],
    'Offensive_xG': [shots_df[shots_df['team'] == 'Argentina']['xG'].sum(),
                     shots_df[shots_df['team'] == 'France']['xG'].sum()],
    'Defensive_xG': [shots_df[shots_df['team'] == 'France']['xG'].sum(),
                     shots_df[shots_df['team'] == 'Argentina']['xG'].sum()]
})
teams_df['xG_Differential'] = teams_df['Offensive_xG'] - teams_df['Defensive_xG']

# Additional Metrics
passes = events[events['type'] == 'Pass']
pressures = events[events['type'] == 'Pressure']
duels = events[events['type'] == 'Duel']
carries = events[events['type'] == 'Carry']
for team in teams_df['Team']:
    team_passes = passes[passes['team'] == team]
    teams_df.loc[teams_df['Team'] == team, 'Pass_Success'] = team_passes['pass_outcome'].isna().mean()
    teams_df.loc[teams_df['Team'] == team, 'Possession'] = (events['possession_team'] == team).mean()
    teams_df.loc[teams_df['Team'] == team, 'Duel_Success'] = duels[duels['team'] == team]['duel_outcome'].notna().mean()
    teams_df.loc[teams_df['Team'] == team, 'Pressure_Count'] = pressures[pressures['team'] == team].shape[0]


# Tactical Suggestions with Visuals
def ai_coach_suggestion(team, minute=45):
    team_data = teams_df[teams_df['Team'] == team].iloc[0]
    opp_team = 'France' if team == 'Argentina' else 'Argentina'
    opp_data = teams_df[teams_df['Team'] == opp_team].iloc[0]
    pitch = Pitch(pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))

    # 1. Attack (Push Flank)
    if team_data['xG_Differential'] > 0.5 and team_data['Possession'] > 0.55:
        shot_locs = shots_df[shots_df['team'] == team]['location']
        flank = "left" if shot_locs.apply(lambda x: x[1]).mean() < 40 else "right"
        sns.kdeplot(x=shot_locs.apply(lambda x: x[0]), y=shot_locs.apply(lambda x: x[1]), fill=True, cmap='Reds', ax=ax)
        pitch.arrows(60, 20 if flank == 'left' else 60, 100, 20 if flank == 'left' else 60, ax=ax, color='blue')
        plt.title(f"{team}: Push {flank} Flank")
        suggestion = f"{team}: Push the {flank} flank—xG diff {team_data['xG_Differential']:.2f}, possession at {team_data['Possession']:.2f}."

    # 2. Defense (Drop Back)
    elif team_data['xG_Differential'] < -0.5 and minute > 75:
        opp_shots = shots_df[shots_df['team'] == opp_team]['location']
        weak_zone = "left" if opp_shots.apply(lambda x: x[1]).mean() < 40 else "right"
        sns.kdeplot(x=opp_shots.apply(lambda x: x[0]), y=opp_shots.apply(lambda x: x[1]), fill=True, cmap='Reds', ax=ax)
        pitch.lines(0, 20 if weak_zone == 'left' else 60, 40, 20 if weak_zone == 'left' else 60, ax=ax, color='yellow',
                    lw=3)
        plt.title(f"{team}: Defend {weak_zone} Zone")
        suggestion = f"{team}: Drop back—xG diff {team_data['xG_Differential']:.2f}. Mark their {weak_zone} attack late game."

    # 3. Pressing (High Press)
    elif team_data['Pass_Success'] > 0.85 and opp_data['Pass_Success'] < 0.7:
        opp_pressures = pressures[pressures['team'] == opp_team]['location']
        pitch.scatter(opp_pressures.apply(lambda x: x[0]), opp_pressures.apply(lambda x: x[1]), ax=ax, c='red', s=50)
        plt.title(f"{team}: Press High")
        suggestion = f"{team}: Press high—opponent’s pass success down to {opp_data['Pass_Success']:.2f}."

    # 4. Substitution (Fatigue)
    elif minute > 60 and team_data['Pass_Success'] < 0.75:
        plt.close(fig)  # Switch to line plot
        pass_time = passes[passes['team'] == team].groupby('minute')['pass_outcome'].apply(lambda x: x.isna().mean())
        plt.figure(figsize=(10, 6))
        plt.plot(pass_time.index, pass_time, marker='o')
        plt.axvline(minute, color='red', linestyle='--')
        plt.title(f"{team}: Pass Success Over Time")
        plt.xlabel("Minute")
        plt.ylabel("Pass Success Rate")
        suggestion = f"{team}: Sub a midfielder—pass success dropped to {team_data['Pass_Success']:.2f} after {minute} mins."

    # 5. Counter-Attack
    elif opp_data['Possession'] > 0.6 and opp_data['Pressure_Count'] > team_data['Pressure_Count'] * 1.5:
        opp_pressures = pressures[pressures['team'] == opp_team]['location']
        sns.kdeplot(x=opp_pressures.apply(lambda x: x[0]), y=opp_pressures.apply(lambda x: x[1]), fill=True,
                    cmap='Reds', ax=ax)
        pitch.arrows(20, 40, 100, 40, ax=ax, color='blue')
        plt.title(f"{team}: Counter-Attack")
        suggestion = f"{team}: Counter-attack now—opponent overcommitting with {opp_data['Pressure_Count']} pressures."

    # 6. Set-Piece Focus
    elif len(set_piece_shots := shots_df[shots_df['shot_type'].isin(['Free Kick', 'Corner'])]) > 0 and \
            set_piece_shots[set_piece_shots['team'] == team]['xG'].sum() > 0.5:
        set_locs = set_piece_shots[set_piece_shots['team'] == team]['location']
        pitch.scatter(set_locs.apply(lambda x: x[0]), set_locs.apply(lambda x: x[1]), ax=ax, c='yellow', s=100)
        plt.title(f"{team}: Set-Piece Focus")
        suggestion = f"{team}: Focus on set pieces—xG from set plays at {set_piece_shots[set_piece_shots['team'] == team]['xG'].sum():.2f}."

    # 7. Player Marking
    elif not (opp_shots := shots_df[shots_df['team'] == opp_team]).empty:
        top_scorer = opp_shots.loc[opp_shots['xG'].idxmax(), 'player']
        top_shots = opp_shots[opp_shots['player'] == top_scorer]['location']
        pitch.scatter(top_shots.apply(lambda x: x[0]), top_shots.apply(lambda x: x[1]), ax=ax, c='red', s=100)
        plt.title(f"{team}: Mark {top_scorer}")
        suggestion = f"{team}: Mark {top_scorer}—their top threat with {opp_shots['xG'].max():.2f} xG."

    # 8. Formation Switch
    elif team_data['Duel_Success'] < 0.5 and team_data['Possession'] < 0.45:
        plt.close(fig)  # Switch to bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Team', y='Duel_Success', data=teams_df, palette='viridis')
        plt.title(f"{team}: Duel Success Comparison")
        plt.ylabel("Duel Success Rate")
        suggestion = f"{team}: Switch to 4-4-2—duels lost ({team_data['Duel_Success']:.2f}), possession low at {team_data['Possession']:.2f}."

    # 9. Wing Play
    elif (opp_shots := shots_df[shots_df['team'] == opp_team]['location'].apply(lambda x: x[1])).between(20,
                                                                                                         60).mean() > 0.7:
        sns.kdeplot(x=shots_df[shots_df['team'] == opp_team]['location'].apply(lambda x: x[0]),
                    y=shots_df[shots_df['team'] == opp_team]['location'].apply(lambda x: x[1]), fill=True, cmap='Reds',
                    ax=ax)
        pitch.arrows(60, 10, 100, 10, ax=ax, color='blue')  # Left wing
        pitch.arrows(60, 70, 100, 70, ax=ax, color='blue')  # Right wing
        plt.title(f"{team}: Exploit Wings")
        suggestion = f"{team}: Exploit the wings—opponent shots central ({opp_shots.between(20, 60).mean():.2f} ratio)."

    # Default
    else:
        sns.kdeplot(x=shots_df[shots_df['team'] == team]['location'].apply(lambda x: x[0]),
                    y=shots_df[shots_df['team'] == team]['location'].apply(lambda x: x[1]), fill=True, cmap='Reds',
                    ax=ax)
        plt.title(f"{team}: Hold Steady")
        suggestion = f"{team}: Hold steady—xG diff {team_data['xG_Differential']:.2f}, possession {team_data['Possession']:.2f}."

    plt.savefig(os.path.expanduser("~/tactic_plot.png"))
    plt.close()
    return suggestion


# Interactive Loop
print("Welcome, 2026 Coach! Type 'team minute' (e.g., 'Argentina 45') or 'quit'.")
while True:
    command = input("Your command: ").strip()
    if command.lower() == 'quit':
        break
    try:
        team, minute = command.split(maxsplit=1)
        minute = int(minute)
        if team in teams_df['Team'].values:
            print(ai_coach_suggestion(team, minute))
            print("Check ~/tactic_plot.png for visual!")
        else:
            print("Team not found. Try Argentina or France.")
    except ValueError:
        print("Format: 'team minute' (e.g., 'Argentina 45')")