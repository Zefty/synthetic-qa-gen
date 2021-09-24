import csv
import os
import json
import re
from tqdm import tqdm 

directory = r'synthetic-qa-gen/data'
for filename in tqdm(sorted(os.listdir(directory))):
    fileNameSplit = re.split('[_.]', filename)
    with open('{}/{}'.format(directory, filename), 'r') as infile:
        with open('{}.{}'.format(fileNameSplit[0], fileNameSplit[-1]), 'a') as outfile:
            outfile.write(infile.read())
            outfile.write("\n")
  
