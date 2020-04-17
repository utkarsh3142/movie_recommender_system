# Get in-data movie suggestions using saved model files

# Import dependencies
import pickle
import numpy as np
# from keras.models import load_model

# Load mapping dictionaries
with open('model/movie_to_idx.pickle', 'rb') as handle:
    movie_to_idx = pickle.load(handle)

with open('model/link_to_idx.pickle', 'rb') as handle:
    link_to_idx = pickle.load(handle)

with open('model/idx_to_movie.pickle', 'rb') as handle:
    idx_to_movie = pickle.load(handle)

# Load movie encodings
movie_encodings = np.load('model/movie_encodings.npy')

# Load encoder
# encoder = load_model('model/encoder.h5')
# print(encoder.summary())

# Movie list
movies = movie_to_idx.keys()

# Get similar movies using the embeddings
def getSimilarMovies(movie_name, similar_num):
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

print(getSimilarMovies('Deadpool (film)',10))