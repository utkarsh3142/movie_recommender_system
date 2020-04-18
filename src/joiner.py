from multiprocessing.dummy import Pool as Threadpool
from itertools import chain
import os
from timeit import default_timer as timer
import json

# Current working directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

partition_data_dir = curr_dir + '/../data/partitions/'
combined_data_file = curr_dir + '/../data/wp_movies.json'

def read_data(file_path):
    """Read in json data from `file_path`"""
    
    data = []
    
    # Open the file and load in json
    with open(file_path, 'r') as fin:
        for l in fin.readlines():
            data.append(json.loads(l))
            
    return data


start = timer()

# List of files to read in
saved_files = [ partition_data_dir + x for x in os.listdir(partition_data_dir)]

# Create a threadpool for reading in files
threadpool = Threadpool(processes = 12)

# Read in the files as a list of lists
results = threadpool.map(read_data, saved_files)

# Flatten the list of lists to a single list
book_list = list(chain(*results))

end = timer()

print(f'Found {len(book_list)} books in {round(end - start)} seconds.')

if not os.path.exists(os.getcwd() + combined_data_file ):
    with open(combined_data_file, 'wt') as fout:
        for book in book_list:
             fout.write(json.dumps(book) + '\n')
    print('Books saved.')
else:
    print('Files already saved.')
