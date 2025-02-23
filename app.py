from flask import Flask, render_template
from heat_map_players import app as dash_app  # Import Dash app

app = Flask(__name__)

# Attach Dash app to Flask
dash_app.server = app
dash_app.config.suppress_callback_exceptions = True  # Suppress callback errors

@app.route('/')
def home():
    features = [
        {"title": "Pre Match Analysis", "description": "Unleash data magic in a flash."},
        {"title": "Live Match Analysis", "description": "Link up like never before."},
        {"title": "Post Match Analysis", "description": "AI that works harder than you."}
    ]
    return render_template('index.html', features=features)

@app.route('/pre-analysis/')
def dashboard():
    return dash_app.index()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)