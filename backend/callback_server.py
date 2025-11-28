# callback_server.py
from flask import Flask, request

app = Flask(__name__)

@app.route("/callback")
def callback():
    auth_code = request.args.get("code")
    return f"AUTH_CODE: {auth_code}"

if __name__ == "__main__":
    app.run(port=8501)