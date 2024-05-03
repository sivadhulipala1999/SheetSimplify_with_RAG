"""Flask web server exposing endpoints to LLM chats."""
import os

from flask import Flask, request, jsonify
from model import llm

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Welcome to Sheet Simplify!"


@app.route("/v1/summary", methods=["GET", "POST"])
def summary():
    """Provide a summary of the data provided. Responds to both GET and POST requests."""
    return llm.main("dev", API_CALL=True)


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
