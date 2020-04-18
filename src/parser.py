from timeit import default_timer as timer
import xml.sax
import mwparserfromhell
from keras.utils import get_file
import bz2
import subprocess
import requests
from bs4 import BeautifulSoup
import re
import gc
import json
from multiprocessing import Pool
import tqdm
from itertools import chain
from functools import partial
import os

# Current working directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

download_path = curr_dir + '/../data/datasets/'
partition_dir = curr_dir + '/../data/partitions/'
downloaded_files_path = [ download_path + file_name for file_name in os.listdir(download_path) ]

# Define number of processes
num_process = 2

### THIS FUNCTION IS TAKEN FROM Deep Learning Cookbook by Douwe Osinga
def process_article(title, text, timestamp, template = 'Infobox film'):
    """Process a wikipedia article looking for template"""
    
    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)
    
    # Search through templates for the template
    matches = wikicode.filter_templates(matches = template)
    
    # Filter out errant matches
    matches = [x for x in matches if x.name.strip_code().strip().lower() == template.lower()]
    
    if len(matches) >= 1:
        # template_name = matches[0].name.strip_code().strip()

        # Extract information from infobox
        properties = {param.name.strip_code().strip(): param.value.strip_code().strip() 
                      for param in matches[0].params
                      if param.value.strip_code().strip()}

        # Extract internal wikilinks
        wikilinks = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]

        # Extract external links
        exlinks = [x.url.strip_code().strip() for x in wikicode.filter_external_links()]

        # Find approximate length of article
        text_length = len(wikicode.strip_code().strip())

        return (title, properties, wikilinks, exlinks, timestamp, text_length)

### THIS CLASS IS BASED ON Deep Learning Cookbook by Douwe Osinga
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Parse through XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._movies = []
        self._article_count = 0
        self._non_matches = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._article_count += 1
            # Search through the page to see if the page is a movie
            movie = process_article(**self._values, template = 'Infobox film')
            # Append to the list of movie
            if movie:
                self._movies.append(movie)

### THIS FUNCTION is taken from Will Koehrsen's project https://github.com/WillKoehrsen/wikipedia-data-science
def find_movies(data_path, limit = None, save = True):

    # Object for handling xml
    handler = WikiXmlHandler()
    
    print("Working on --> ", data_path)

    # Parsing object
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    # Iterate through compressed file
    for i, line in enumerate(subprocess.Popen(['bzcat'], 
                             stdin = open(data_path), 
                             stdout = subprocess.PIPE).stdout):
        try:
            parser.feed(line)
        except StopIteration:
            break
            
        # Optional limit
        if limit is not None and len(handler._movies) >= limit:
            return handler._movies
    
    if save:
        # Create file name based on partition name
        p_str = data_path.split('-')[-1].split('.')[-2]
        out_dir = partition_dir + f'{p_str}.ndjson'

        # Open the file
        with open(out_dir, 'w') as fout:
            # Write as json
            for movie in handler._movies:
                fout.write(json.dumps(movie) + '\n')
        
        print(f'{len(os.listdir(partition_dir))} files processed.', end = '\r')

    # Memory management
    del handler
    del parser
    gc.collect()
    return None


# Create a pool of workers to execute processes
pool = Pool(processes = num_process)

start = timer()

# Map (service, tasks), applies function to each partition
#results = pool.map(find_books, downloaded_files_path[0:4])

results = []

# # Run partitions in parallel
for x in tqdm.tqdm(pool.imap_unordered(find_movies, downloaded_files_path ), total = len(downloaded_files_path)):
     results.append(x)
    

pool.close()
pool.join()

end = timer()
print(f'{end - start} seconds elapsed.')


