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
    summary_question = "Give me a summary of the data provided"
    llm_chain = llm.setup("dev")
    return llm_chain.invoke(summary_question).split("assistant")[1]


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
