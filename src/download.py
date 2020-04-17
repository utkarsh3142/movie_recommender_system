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

# Define the dump date index

dump_date = '20200401'

base_url = 'https://dumps.wikimedia.org/enwiki/'
download_path = curr_dir + '/../data/'
dumps_url = base_url + dump_date + '/'


def generate_dumps_list(base_url, dump_date):
    
    files = []
    
    if not base_url.endswith('/'):
        dumps_url = base_url + '/' + dump_date + '/'
    else:
        dumps_url = base_url + dump_date + '/'
        
    print("Dump URL --> ", dumps_url)
       
    try:
        dump_html = requests.get(dumps_url).text
    except requests.exceptions.RequestException as e:  
        raise SystemExit(e)
        
    soup_dump = BeautifulSoup(dump_html, 'html.parser')

    for file in soup_dump.find_all('li', {'class': 'file'}):
        text = file.text
        if 'pages-articles' in text and  '.xml-p' in text and not 'multistream' in text:
            files.append(text.split()[0])
            
    return files


def download_files(dumps_list, download_path, dumps_url, limit = None):
    
    data_paths = []
    
    start = timer()
    for indx, file in enumerate(dumps_list):
        if limit != None and limit < indx:
            break
        else:
            data_paths.append(get_file(cache_dir=download_path, fname=file, origin=dumps_url + file))
    end = timer()
    print(f'{round(end - start)} total seconds elapsed.')
    
    return data_paths


wikidumps_list = generate_dumps_list(base_url, dump_date)

downloaded_files_path = download_files(wikidumps_list, download_path, dumps_url)

print(downloaded_files_path) 

