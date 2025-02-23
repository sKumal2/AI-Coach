from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Eye-catching titles and short descriptions
    features = [
        {"title": "Pre Modal", "description": "Unleash data magic in a flash."},
        {"title": "Live Modal", "description": "Link up like never before."},
        {"title": "Post Modal", "description": "AI that works harder than you."}
    ]
    return render_template('index.html', features=features)

@app.route('/app')
def app_page():
    return render_template('app_page.html')

if __name__ == "__main__":
    app.run(debug=True)