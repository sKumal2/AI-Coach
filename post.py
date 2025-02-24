import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('Agg')  # Thread-safe backend

import matplotlib.pyplot as plt

import seaborn as sns

from mplsoccer import Pitch

from statsbombpy import sb

import os

from flask import Flask, render_template, send_from_directory

 

app = Flask(__name__)

 

# Ensure static folder exists

if not os.path.exists('static'):

    os.makedirs('static')

 

# Fetch StatsBomb Data (2022 WC Final)

events = sb.events(match_id=3869685)

shots_df = events[events['type'] == 'Shot'].copy()

passes = events[events['type'] == 'Pass'].copy()

pressures = events[events['type'] == 'Pressure']

duels = events[events['type'] == 'Duel']

 

# Enhanced xG Features

shots_df['distance'] = np.sqrt((120 - shots_df['location'].apply(lambda x: x[0])) ** 2 +

                               (40 - shots_df['location'].apply(lambda x: x[1])) ** 2)

shots_df['angle'] = np.arctan2(40 - shots_df['location'].apply(lambda x: x[1]),

                               120 - shots_df['location'].apply(lambda x: x[0])).abs() * 180 / np.pi

shots_df['goal'] = shots_df['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

shots_df['is_header'] = shots_df['shot_body_part'].apply(lambda x: 1 if x == 'Head' else 0)

shots_df['is_open_play'] = shots_df['shot_type'].apply(lambda x: 1 if x == 'Open Play' else 0)

shots_df['under_pressure'] = shots_df['under_pressure'].fillna(False).astype(int, copy=False)

 

def goalkeeper_distance(row):

    if isinstance(row['shot_freeze_frame'], list):

        for player in row['shot_freeze_frame']:

            if player['position'] == 'Goalkeeper':

                gk_pos = player['location']

                shot_pos = row['location']

                return np.sqrt((shot_pos[0] - gk_pos[0])**2 + (shot_pos[1] - gk_pos[1])**2)

    return np.nan

shots_df['gk_distance'] = shots_df.apply(goalkeeper_distance, axis=1)

shots_df['gk_distance'] = shots_df['gk_distance'].fillna(shots_df['gk_distance'].mean())

 

def defender_count(row, radius=5):

    if isinstance(row['shot_freeze_frame'], list):

        shot_pos = row['location']

        defenders = [p for p in row['shot_freeze_frame'] if not p['teammate'] and p['position'] != 'Goalkeeper']

        return sum(1 for d in defenders if np.sqrt((shot_pos[0] - d['location'][0])**2 + (shot_pos[1] - d['location'][1])**2) <= radius)

    return 0

shots_df['defender_density'] = shots_df.apply(defender_count, axis=1)

 

shots_df['is_volley'] = shots_df['shot_technique'].apply(lambda x: 1 if x == 'Volley' else 0)

shots_df['is_big_chance'] = shots_df['shot_key_pass_id'].notna().astype(int)

 

# Ensure no NaNs

features = ['distance', 'angle', 'is_header', 'is_open_play', 'under_pressure', 'gk_distance', 'defender_density', 'is_volley', 'is_big_chance']

for feature in features:

    shots_df[feature] = shots_df[feature].fillna(0)

 

# Train xG Model

X = shots_df[features]

y = shots_df['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

shots_df['xG'] = model.predict_proba(X)[:, 1]

print(f"Enhanced xG Model Accuracy: {model.score(X_test, y_test):.2f}")

 

# Precompute per-minute stats

teams = ['Argentina', 'France']

per_minute_stats = {team: {'xG': [], 'Possession': [], 'Pass_Success': []} for team in teams}

for minute in range(1, 121):

    events_up_to_minute = events[events['minute'] <= minute]

    shots_up_to_minute = shots_df[shots_df['minute'] <= minute]

   

    for team in teams:

        # xG up to this minute

        team_xG = shots_up_to_minute[shots_up_to_minute['team'] == team]['xG'].sum()

        opp_xG = shots_up_to_minute[shots_up_to_minute['team'] != team]['xG'].sum()

        xG_diff = team_xG - opp_xG

        per_minute_stats[team]['xG'].append(xG_diff)

       

        # Possession (approximated from possession_team events)

        possession_events = events_up_to_minute[events_up_to_minute['possession_team'].notna()]

        possession_pct = (possession_events['possession_team'] == team).mean() * 100 if not possession_events.empty else 50.0

        per_minute_stats[team]['Possession'].append(possession_pct)

       

        # Pass Success (5-minute window)

        recent_passes = passes[(passes['minute'] > max(0, minute - 5)) & (passes['minute'] <= minute)]

        pass_success = recent_passes[recent_passes['team'] == team]['pass_outcome'].isna().mean() * 100 if not recent_passes.empty else 0.0

        per_minute_stats[team]['Pass_Success'].append(pass_success)

 

def ai_coach_suggestion(team, minute):

    events_up_to_minute = events[events['minute'] <= minute]

    shots_up_to_minute = shots_df[shots_df['minute'] <= minute]

    opp_team = 'France' if team == 'Argentina' else 'Argentina'

 

    # Dynamic stats

    team_xG = shots_up_to_minute[shots_up_to_minute['team'] == team]['xG'].sum()

    opp_xG = shots_up_to_minute[shots_up_to_minute['team'] == opp_team]['xG'].sum()

    xG_diff = team_xG - opp_xG

    possession = per_minute_stats[team]['Possession'][minute - 1]

    pass_success = per_minute_stats[team]['Pass_Success'][minute - 1]

    opp_pass_success = per_minute_stats[opp_team]['Pass_Success'][minute - 1]

    pressure_count = pressures[pressures['team'] == team].shape[0]

 

    recent_shots = shots_df[(shots_df['minute'] > max(0, minute - 5)) & (shots_df['minute'] <= minute)]

    team_xG_recent = recent_shots[recent_shots['team'] == team]['xG'].sum()

    opp_xG_recent = recent_shots[recent_shots['team'] == opp_team]['xG'].sum()

 

    # Setup figure

    fig = plt.figure(figsize=(14, 7))

    ax_pitch = fig.add_axes([0.05, 0.05, 0.65, 0.9])

    ax_stats = fig.add_axes([0.75, 0.05, 0.2, 0.9])

    pitch = Pitch(pitch_color='grass', line_color='white')

    pitch.draw(ax=ax_pitch)

 

    stats = {

        'xG Diff': xG_diff,

        'Possession %': possession,

        'Pass Success %': pass_success,

        'Avg xG/Shot': shots_up_to_minute[shots_up_to_minute['team'] == team]['xG'].mean() if not shots_up_to_minute[shots_up_to_minute['team'] == team].empty else 0.0,

        'Pressure Count': pressure_count

    }

 

    # Tactical Scenarios

    if xG_diff > 0.5 and possession > 55:

        shot_locs = shots_up_to_minute[shots_up_to_minute['team'] == team]['location']

        flank = "left" if shot_locs.apply(lambda x: x[1]).mean() < 40 else "right"

        sns.kdeplot(x=shot_locs.apply(lambda x: x[0]), y=shot_locs.apply(lambda x: x[1]), fill=True, cmap='Reds', ax=ax_pitch)

        pitch.arrows(60, 20 if flank == 'left' else 60, 100, 20 if flank == 'left' else 60, ax=ax_pitch, color='blue')

        ax_pitch.set_title(f"{team}: Push {flank} Flank")

        suggestion = f"Push the {flank} flank—xG diff {stats['xG Diff']:.2f}, possession {stats['Possession %']:.1f}%."

 

    elif xG_diff < -0.5 and minute > 75:

        opp_shots = shots_up_to_minute[shots_up_to_minute['team'] == opp_team]['location']

        weak_zone = "left" if opp_shots.apply(lambda x: x[1]).mean() < 40 else "right"

        sns.kdeplot(x=opp_shots.apply(lambda x: x[0]), y=opp_shots.apply(lambda x: x[1]), fill=True, cmap='Reds', ax=ax_pitch)

        pitch.lines(0, 20 if weak_zone == 'left' else 60, 40, 20 if weak_zone == 'left' else 60, ax=ax_pitch, color='yellow', lw=3)

        ax_pitch.set_title(f"{team}: Defend {weak_zone} Zone")

        suggestion = f"Drop back to defend {weak_zone}—xG diff {stats['xG Diff']:.2f}, opp recent xG {opp_xG_recent:.2f}."

 

    elif pass_success > 85 and opp_pass_success < 70:

        opp_pressures = pressures[pressures['team'] == opp_team]['location']

        pitch.scatter(opp_pressures.apply(lambda x: x[0]), opp_pressures.apply(lambda x: x[1]), ax=ax_pitch, c='red', s=50)

        ax_pitch.set_title(f"{team}: Press High")

        suggestion = f"Press high—pass success {stats['Pass Success %']:.1f}% vs. {opp_team}'s {opp_pass_recent*100:.1f}%."

 

    else:

        sns.kdeplot(x=shots_up_to_minute[shots_up_to_minute['team'] == team]['location'].apply(lambda x: x[0]),

                    y=shots_up_to_minute[shots_up_to_minute['team'] == team]['location'].apply(lambda x: x[1]), fill=True, cmap='Reds', ax=ax_pitch)

        ax_pitch.set_title(f"{team}: Hold Steady")

        suggestion = f"Hold steady—xG diff {stats['xG Diff']:.2f}, recent xG {team_xG_recent:.2f}."

 

    # Sidebar stats

    ax_stats.axis('off')

    stat_text = "\n".join([f"{k}: {v:.1f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()])

    ax_stats.text(0, 0.5, stat_text, fontsize=12, ha='left', va='center')

 

    filename = f"tactic_plot_{team}_{minute}.png"

    plt.savefig(os.path.join('static', filename))

    plt.close()

 

    return {

        'suggestion': suggestion,

        'stats': stats,

        'plot_url': f"/static/{filename}"

    }

 

@app.route('/')

def main():

    intervals = range(1, 121)  # Every minute from 1 to 120

    return render_template('main.html', intervals=intervals)

 

@app.route('/insights/<int:minute>')

def insights(minute):

    if minute < 1 or minute > 120:

        return "Invalid minute. Use values between 1 and 120.", 400

 

    argentina_insight = ai_coach_suggestion('Argentina', minute)

    france_insight = ai_coach_suggestion('France', minute)

 

    return render_template('insights.html', minute=minute, argentina=argentina_insight, france=france_insight)

 

@app.route('/static/<path:filename>')

def serve_static(filename):

    return send_from_directory('static', filename)

 

if __name__ == '__main__':

    app.run(debug=True, threaded=False)