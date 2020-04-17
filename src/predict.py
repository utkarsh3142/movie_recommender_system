# Get in-data movie suggestions using saved model files

# Import dependencies
import pickle
import numpy as np
# from keras.models import load_model
import os

# Current working directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Directories path for src, model and data
src_dir = curr_dir + '/../src'
model_dir = curr_dir + '/../model'
data_dir = curr_dir + '/../data'

# Model files path
movie_idx_file = model_dir + '/movie_to_idx.pickle'
link_idx_file = model_dir + '/link_to_idx.pickle'
idx_movie_file = model_dir + '/idx_to_movie.pickle'
movie_encoding_file = model_dir + '/movie_encodings.npy'
encoder_model_file = model_dir + '/encoder.h5'


# Load mapping dictionaries
with open(movie_idx_file, 'rb') as handle:
    movie_to_idx = pickle.load(handle)

with open(link_idx_file, 'rb') as handle:
    link_to_idx = pickle.load(handle)

with open(idx_movie_file, 'rb') as handle:
    idx_to_movie = pickle.load(handle)

# Load movie encodings
movie_encodings = np.load(movie_encoding_file)

# Load encoder
# encoder = load_model(cfg.encoder_model_file)
# print(encoder.summary())

# Movie list
movies = movie_to_idx.keys()

# Get similar movies using the embeddings
def getSimilarMovies(movie_name, similar_num=10):
  movie_idx = movie_to_idx[movie_name]
  movie_encoding = movie_encodings[movie_idx]
  
  # Calculate similarities with rest of the movies
  similarities = []
  for i in range(len(movies)):
    if i != movie_idx:
      similarities.append((i,np.square(movie_encodings[i] - movie_encoding).mean(axis = 0)))
  
    # Sort by similarities  
  similarities = sorted(similarities, key = lambda x: x[1])

  # Take top similar_num matches
  similar_movies = []
  for i in similarities[:similar_num]:
    similar_movies.append(idx_to_movie[i[0]])

  return similar_movies

#print(getSimilarMovies('Deadpool (film)',10))
