from flask import Flask, render_template, request, jsonify
from app2 import pipeline
import threading


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

import os
import time

def run_pipeline():
    file_path = "recording.wav"
    
    while True:
        if os.path.exists(file_path):  # Check if the file exists
            pipeline(file_path)  # Process the file
            os.remove(file_path)  # Optionally remove the file after processing
        else:
            time.sleep(1)

threading.Thread(target=run_pipeline, daemon=True).start()
    
if __name__ == "__main__":
    app.run(debug=True,port=8001)
