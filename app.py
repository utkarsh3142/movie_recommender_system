import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from src.predict import *

# app
app = Flask(__name__)

# routes

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/recommend', methods=['POST','GET'])
def predict():
    
    # get data
    data = request.get_json(force=True)
    
    #print(data)

    movie = data['movie']

    if 'number' in data:
        similar_count = data['number']
    
        # predictions
        result = getSimilarMovies(movie_name=movie, similar_num=similar_count)
    else:
        result = getSimilarMovies(movie_name=movie)

    # send back to browser
    output = {"recommendations":result}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
# load model
    
    app.run(port = 5000, debug=True)
