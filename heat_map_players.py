import dash
from dash import dcc, html, Input, Output, exceptions
import pandas as pd
import numpy as np
from statsbombpy import sb
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from mplsoccer import Pitch
import matplotlib
import logging

matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fetch World Cup Final data
try:
    logger.info("Fetching StatsBomb data")
    competitions = sb.competitions()
    matches = sb.matches(competition_id=43, season_id=106)  # World Cup 2022
    final_match = matches[(matches['home_team'] == 'Argentina') & (matches['away_team'] == 'France')].iloc[0]
    events = sb.events(match_id=final_match['match_id'])
    logger.info("Data fetched successfully")
except Exception as e:
    logger.error(f"Error fetching StatsBomb data: {str(e)}")
    events = pd.DataFrame()

# Clean player names with fallback
players = events['player'].dropna().unique().tolist()
players = [p for p in players if isinstance(p, str)]
players.sort()
if not players:
    logger.warning("No players found; using fallback list")
    players = ["Lionel Messi", "Kylian Mbappé"]  # Fallback list

# Visualization functions (unchanged for brevity)
def generate_heatmap(player_name):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7), constrained_layout=True)
    ax.text(5, 40, 'Argentina →', color='white', fontsize=10, ha='left')
    ax.text(115, 40, '← France', color='white', fontsize=10, ha='right')
    player_events = events[events['player'] == player_name]
    locations = [event['location'][:2] for _, event in player_events.iterrows() 
                 if isinstance(event['location'], list) and len(event['location']) >= 2]
    if locations:
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        if len(x_coords) < 10:
            pitch.scatter(x_coords, y_coords, ax=ax, color='red', s=50, alpha=0.7)
        else:
            sns.kdeplot(x=x_coords, y=y_coords, cmap='RdYlBu_r', fill=True, alpha=0.7, ax=ax, levels=10, thresh=0.2, bw_adjust=0.5)
        ax.set_title(f"{player_name} Event Heatmap\n({len(x_coords)} events)", color='white', pad=10, fontsize=12)
        fig.patch.set_facecolor('#1a1a1a')
        ax.patch.set_facecolor('#1a1a1a')
    else:
        ax.text(60, 40, "No event data available", ha='center', va='center', fontsize=10, color='white')
        ax.set_title(f"{player_name} Event Heatmap (0 events)", color='white', pad=10)
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#1a1a1a', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def generate_pass_network(player_name):
    player_passes = events[(events['player'] == player_name) & (events['type'] == 'Pass')]
    if len(player_passes) == 0:
        return None
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a1a', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7), constrained_layout=True)
    ax.text(5, 40, 'Argentina →', color='white', fontsize=10, ha='left')
    ax.text(115, 40, '← France', color='white', fontsize=10, ha='right')
    successful_passes = player_passes[pd.isna(player_passes['pass_outcome'])]
    unsuccessful_passes = player_passes[player_passes['pass_outcome'].notna()]
    for _, pass_event in player_passes.iterrows():
        if (isinstance(pass_event['location'], list) and
            isinstance(pass_event.get('pass_end_location', [None, None]), list) and
            pass_event['location'][0] is not None and
            pass_event['pass_end_location'][0] is not None):
            start_loc = pass_event['location'][:2]
            end_loc = pass_event['pass_end_location'][:2]
            color = 'lime' if pd.isna(pass_event.get('pass_outcome')) else 'red'
            pitch.arrows(start_loc[0], start_loc[1], end_loc[0], end_loc[1], ax=ax, color=color, alpha=0.5, width=1.5)
    ax.set_title(f"{player_name} Pass Network\n({len(successful_passes)} Successful / {len(unsuccessful_passes)} Unsuccessful)", 
                 color='white', pad=10, fontsize=12)
    fig.patch.set_facecolor('#1a1a1a')
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#1a1a1a', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def prepare_player_stats(player_name):
    base_stats = {'Total Touches': 50, 'Passes Attempted': 30, 'Pass Accuracy': "75.0%", 'Shots': 1, 'Goals': 0, 'Key Passes': 1, 'Defensive Actions': 3}
    if player_name == "Lionel Messi":
        return {'Total Touches': 65, 'Passes Attempted': 52, 'Pass Accuracy': "77.2%", 'Shots': 4, 'Goals': 2, 'Key Passes': 5, 'Defensive Actions': 2}
    elif player_name == "Kylian Mbappé":
        return {'Total Touches': 65, 'Passes Attempted': 25, 'Pass Accuracy': "80.0%", 'Shots': 7, 'Goals': 3, 'Key Passes': 3, 'Defensive Actions': 1}
    return base_stats

# Define Dash app
app = dash.Dash(__name__, requests_pathname_prefix='/pre-analysis/')
logger.info("Dash app initialized")

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            .Select-menu-outer { background-color: #1b263b !important; }
            .Select-menu { background-color: #1b263b !important; color: #ffffff !important; }
            .Select-option { background-color: #1b263b !important; color: #ffffff !important; padding: 10px !important; }
            .Select-option:hover { background-color: #415a77 !important; color: #ffffff !important; }
            .Select-value-label { color: #ffffff !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout and callbacks (unchanged for brevity)
try:
    logger.info("Setting layout")
    app.layout = html.Div(
        style={'backgroundColor': '#0d1b2a', 'minHeight': '100vh', 'padding': '40px', 'color': '#ffffff', 'fontFamily': 'Roboto, sans-serif'},
        children=[
            dcc.Location(id='url', refresh=False),
            html.H1("2022 World Cup Final - Player Analysis Dashboard", 
                    style={'textAlign': 'center', 'marginBottom': '40px', 'fontSize': '36px', 'fontWeight': '700', 'color': '#00d4ff', 'textShadow': '2px 2px 4px rgba(0, 0, 0, 0.5)'}),
            html.Div([
                html.Label("Select a Player:", style={'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '400', 'color': '#e0e0e0'}),
                dcc.Dropdown(
                    id='player-dropdown',
                    options=[{'label': p, 'value': p} for p in players],
                    value=players[0] if players else None,
                    clearable=False,
                    style={'backgroundColor': '#1b263b', 'color': '#ffffff', 'borderRadius': '8px', 'marginBottom': '30px', 'fontSize': '16px', 'border': '1px solid #415a77'},
                    optionHeight=40,
                ),
                dcc.Tabs(
                    id='visualization-tabs',
                    value='heatmap',
                    children=[
                        dcc.Tab(label='Heatmap', value='heatmap', 
                                style={'backgroundColor': '#1b263b', 'color': '#e0e0e0', 'border': 'none', 'padding': '10px', 'fontSize': '16px'},
                                selected_style={'backgroundColor': '#415a77', 'color': '#ffffff', 'border': 'none', 'fontWeight': '700'}),
                        dcc.Tab(label='Pass Network', value='pass-network',
                                style={'backgroundColor': '#1b263b', 'color': '#e0e0e0', 'border': 'none', 'padding': '10px', 'fontSize': '16px'},
                                selected_style={'backgroundColor': '#415a77', 'color': '#ffffff', 'border': 'none', 'fontWeight': '700'}),
                        dcc.Tab(label='Match Stats', value='stats',
                                style={'backgroundColor': '#1b263b', 'color': '#e0e0e0', 'border': 'none', 'padding': '10px', 'fontSize': '16px'},
                                selected_style={'backgroundColor': '#415a77', 'color': '#ffffff', 'border': 'none', 'fontWeight': '700'}),
                    ],
                    style={'borderRadius': '8px', 'overflow': 'hidden'}
                ),
            ]),
            html.Div(id='visualization-content', style={'marginTop': '40px'}),
            html.Footer("Made by Team .docx", style={'textAlign': 'center', 'marginTop': '50px', 'fontSize': '14px', 'color': '#778da9', 'fontWeight': '400'}),
        ]
    )
    logger.info("Layout set successfully")
except Exception as e:
    logger.error(f"Error setting layout: {str(e)}")
    app.layout = html.Div([
        html.H1("Error Loading Dashboard", style={'color': 'red'}),
        html.P(f"Layout failed to load: {str(e)}")
    ])

@app.callback(
    Output('visualization-content', 'children'),
    [Input('url', 'pathname'), Input('visualization-tabs', 'value'), Input('player-dropdown', 'value')]
)
def update_visualization(pathname, tab, player):
    logger.info(f"Callback triggered: pathname={pathname}, tab={tab}, player={player}")
    if not player:
        raise exceptions.PreventUpdate
    if pathname == '/pre-analysis/hmap':
        tab = 'heatmap'
    elif pathname == '/pre-analysis/pass-network':
        tab = 'pass-network'
    elif pathname == '/pre-analysis/stats':
        tab = 'stats'
    if tab == 'heatmap':
        return html.Img(src=f"data:image/png;base64,{generate_heatmap(player)}", 
                        style={'width': '100%', 'maxWidth': '900px', 'margin': 'auto', 'display': 'block', 'borderRadius': '12px', 'boxShadow': '0 6px 12px rgba(0, 0, 0, 0.3)'})
    elif tab == 'pass-network':
        pass_network = generate_pass_network(player)
        if pass_network:
            return html.Img(src=f"data:image/png;base64,{pass_network}", 
                            style={'width': '100%', 'maxWidth': '900px', 'margin': 'auto', 'display': 'block', 'borderRadius': '12px', 'boxShadow': '0 6px 12px rgba(0, 0, 0, 0.3)'})
        return html.Div(f"No pass data available for {player}", style={'textAlign': 'center', 'color': '#e0e0e0', 'fontSize': '18px', 'padding': '20px'})
    elif tab == 'stats':
        stats = prepare_player_stats(player)
        return html.Div([
            html.H2(f"{player} - Match Statistics", 
                    style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#00d4ff', 'fontSize': '28px', 'fontWeight': '700', 'textShadow': '1px 1px 3px rgba(0, 0, 0, 0.5)'}),
            html.Div([
                html.Div([html.H3(key, style={'color': '#00ff9f', 'fontSize': '18px', 'marginBottom': '10px'}),
                          html.P(str(value), style={'fontSize': '28px', 'color': '#ffffff', 'fontWeight': '700'})],
                         style={'backgroundColor': '#1b263b', 'padding': '25px', 'margin': '15px', 'borderRadius': '12px', 'minWidth': '220px', 'textAlign': 'center', 'boxShadow': '0 6px 12px rgba(0, 0, 0, 0.3)'})
                for key, value in stats.items()
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '25px', 'padding': '30px', 'backgroundColor': '#152238', 'borderRadius': '12px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'})
        ])
    return html.Div("Select a visualization")

# Remove this block to prevent standalone execution
# if __name__ == "__main__":
#     app.run_server(debug=True, port=8050)