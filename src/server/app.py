from flask import Flask, render_template, jsonify, request
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

@app.route("/greet", methods=["POST"])
def greet():
    data = request.get_json()

    print(f"received data from browser: {data}")
    
    user_name = data.get("name", "").strip()
    if not user_name:
        user_name = "stranger"
    
    result = f"No {user_name}, I am your father"
    return jsonify({"lebron": result})
    
if __name__ == "__main__":
    app.run(debug=True)
