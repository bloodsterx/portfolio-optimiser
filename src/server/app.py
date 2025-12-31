from flask import Flask, render_template, jsonify, request
from ..inference.predict import predict
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    my_tasks = ["Stock Prediction", "Portfolio Optimisation", "Black-Litterman Portfolio"]
    return render_template("index.html", tasks=my_tasks)

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
    return jsonify({"greeting": result})

@app.route("/profile/<username>")
def profile(username):
    print(username, "tried to log in")
    return render_template("profile.html", name=username)

@app.route("/predict_home")
def predict_home():
    return render_template("predict_home.html")

@app.route("/predict", methods=["POST"])
def get_prediction():
    data = request.get_json()

    print(f"User requesting a stock prediction of {data}")
    try:
        forecast = predict("2025-12-29-DL-weights.pt", data)
    except Exception as e:
        print(e)
        raise e

    return jsonify({ "forecast": forecast })

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
