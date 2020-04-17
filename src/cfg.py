# Configuration file

import os

# Current working directory 
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Directories path for src, model and data
src_dir = curr_dir + '/src'
model_dir = curr_dir + '/model'
data_dir = curr_dir + '/data'

# Model files path 
movie_idx_file = model_dir + '/movie_to_idx.pickle'
link_idx_file = model_dir + '/link_to_idx.pickle'
idx_movie_file = model_dir + '/idx_to_movie.pickle'
movie_encoding_file = model_dir + '/movie_encodings.npy'
encoder_model_file = model_dir + '/encoder.h5'

