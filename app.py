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
		text = request.args.get('jsdata')
		
		# Get recommendations
		result = getSimilarMovies(movie_name=text, similar_num=15)
		
		# Get google search queries
		gs = "https://www.google.com/search?q="
		result_links = [gs + res + ' movie' for res in result]
		
		# Render results in table format
		return render_template('recommendations_table.html', suggestions = zip(result, result_links))
		
		
		"""
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
		"""

if __name__ == '__main__':
# load model
    
    app.run(port = 5000, debug=True)
