# Import necessary dependencies
from collections import Counter
import json
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import pickle
import matplotlib.pyplot as plt


# Path of the {movie : [links]} json file
json_src = ""
# Discard links having count < min_count
min_count = 5
# Path to store model files
savepath = "model/"

# Model config
embedding_size = 50
epochs = 50


# Count the occurances of links
with open(json_src) as fin:
    movies = [json.loads(l) for l in fin]

link_counts = Counter()
for movie in movies:
    link_counts.update(movie[2])
link_counts.most_common(10)

# Total number of links
len(link_counts.most_common())

# List of links whose total counts > min_count
freq_links = []
for link in link_counts.most_common():
  if link[1] > min_count:
    freq_links.append(link[0])

print(len(freq_links))

# Create movie and link mappings
link_to_idx = {link: idx for idx, link in enumerate(freq_links)}
movie_to_idx = {movie[0]: idx for idx, movie in enumerate(movies)}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}

# Save the mappings
with open(savepath + 'link_to_idx.pickle', 'wb') as handle:
    pickle.dump(link_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('link mapping saved to disk')

with open(savepath + 'movie_to_idx.pickle', 'wb') as handle:
    pickle.dump(movie_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('movie mapping saved to disk')

with open(savepath + 'idx_to_movie.pickle', 'wb') as handle:
    pickle.dump(idx_to_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('movie reverse mapping saved to disk')

# Create a numpy 0 matrix
movie_link_mat = np.zeros(shape=[len(movies),len(freq_links)])

# Fill the matrix based on link presence in movies
for movie in movies:
  for link in freq_links:
    if link in movie[2]:
      movie_link_mat[movie_to_idx[movie[0]]][link_to_idx[link]] = 1

# Save the matrix
# np.save('/content/movie_link_mat.npy', movie_link_mat)

# Create an AE to get the embeddings
inp_layer = Input(name = 'input', shape=(len(freq_links),))
embedding_layer = Dense(name = 'embeddings', units = embedding_size, activation = 'relu')(inp_layer)
out_layer = Dense(name = 'output', units = len(freq_links), activation = 'softmax')(embedding_layer)

model = Model(inputs = inp_layer, outputs = out_layer, name = 'AutoEncoder')
model.compile(optimizer = 'adam', loss= 'binary_crossentropy')
model.summary()

# Try training the model
history = model.fit(x = movie_link_mat, y = movie_link_mat, epochs = epochs, batch_size = 100, shuffle = True)

# Plot training curves
plt.plot([i for i in range(50)], history.history['loss'])
plt.title('Loss vs epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Set up the encoder
encoder = Model(inputs = inp_layer, outputs = embedding_layer)
encoder.summary()

# Save the encoder
encoder.save(savepath + 'model.h5')
print('Encoder saved to disk')

# Get the encodings for all the movies
movie_encodings = encoder.predict(x = movie_link_mat)

# Save the encodings
np.save(savepath + 'movie_encodings.npy', movie_encodings)
print('Movie encodings saved to disk')

# Get similar movies using the embeddings
def getSimilarMovies(movie_name, similar_num):
  movie_idx = movie_to_idx[movie_name]
  movie_encoding = movie_encodings[movie_idx]
  
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

# getSimilarMovies('Rogue One',10)

"""
Original model
29 Rogue One 0.9999999
3349 Star Wars: The Force Awakens 0.9722805
101 Prometheus (2012 film) 0.9653338
140 Star Trek Into Darkness 0.9635347
22 Jurassic World 0.962336
25 Star Wars sequel trilogy 0.95218825
659 Rise of the Planet of the Apes 0.9516557
62 Fantastic Beasts and Where to Find Them (film) 0.94662267
42 The Avengers (2012 film) 0.94634
37 Avatar (2009 film) 0.9460137
"""