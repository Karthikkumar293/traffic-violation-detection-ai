from flask import Flask, send_from_directory

app = Flask(__name__, static_folder=".")

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return {
        "status": "online",
        "system": "TrafficGuard AI"
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
