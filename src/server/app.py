from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_data")
def get_data():
    time = datetime.now().strftime("%H:%M:%S")
    return jsonify({
        "message": f"Here's the time, thanks for requesting! {time}",
        "status": "success" # non-essential, but professional convention
    })

if __name__ == "__main__":
    app.run(debug=True)
