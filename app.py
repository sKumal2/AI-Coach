from flask import Flask, render_template
import hmap

app = Flask(__name__)

@app.route('/')
def home():
        # Eye-catching titles and short descriptions
        features = [
            {"title": "Pre Match Analysis", "description": "Unleash data magic in a flash."},
            {"title": "Live Match Analysis", "description": "Link up like never before."},
            {"title": "Post Match Analysis", "description": "AI that works harder than you."}
        ]
        return render_template('index.html', features=features)

@app.route('/app')

def app_page():
    return render_template('app_page.html')

if __name__ == "__main__":
    app.run(debug=True)