import csv
import os
import json
import re
from tqdm import tqdm 

directory = r'synthetic-qa-gen/data'
for filename in tqdm(sorted(os.listdir(directory))):
    print(filename)
    fileNameSplit = re.split('[_.]', filename)
    with open('{}/{}'.format(directory, filename), 'r') as infile:
        with open('synthetic-qa-gen/{}.{}'.format(fileNameSplit[0], fileNameSplit[-1]), 'a') as outfile:
            outfile.write(infile.read())
            outfile.write("\n")
  
