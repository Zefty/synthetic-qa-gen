import csv
import os
import json
import re

def CreateRAGTrainingData(indirectory, outdirectory, files):
    for file in files:
        count = 0
        with open('{}/{}'.format(outdirectory, file), 'w', encoding = 'utf8') as outfile:
            with open('{}/{}'.format(indirectory, file), 'r', encoding = 'utf8') as infile:
                    while count < 100000 and infile:
                        inputLine = infile.readline().strip()
                        if len(inputLine) > 1:
                            outfile.write(inputLine)
                            outfile.write('\n')
                            count = count + 1

def CreateRAGTrainingData2(indirectories, num_rows, outdirectory, files):
    for file in files:
        with open('{}/{}'.format(outdirectory, file), 'w', encoding = 'utf8') as outfile:
            for index, indirectory in enumerate(indirectories):
                count = 0
                with open('{}/{}'.format(indirectory, file), 'r', encoding = 'utf8') as infile:
                    while count < num_rows[index] and infile:
                        inputLine = infile.readline().strip()
                        if len(inputLine) > 1:
                            outfile.write(inputLine)
                            outfile.write('\n')
                            count = count + 1

# For processing the Q-covid data (short answers)
indirectory = r'data/input_data/Q-covid'
outdirectory = r'data/output_data/Q-covid-100k'
files = ['train.source', 'train.target']
CreateRAGTrainingData(indirectory, outdirectory, files)

# For processing the Q-covid-summarised data
indirectory = r'data/input_data/Q-covid-summary'
outdirectory = r'data/output_data/Q-covid-summary-100k'
files = ['train.source', 'train.target']
CreateRAGTrainingData(indirectory, outdirectory, files)

# For processing the Q-covid data + Q-covid-summarised data
indirectories = [r'data/input_data/Q-covid', r'data/input_data/Q-covid-summary']
num_rows = [100000, 50000]
outdirectory = r'data/output_data/Q-covid-100k+Q-covid-summary-50k'
files = ['train.source', 'train.target']
CreateRAGTrainingData2(indirectories, num_rows, outdirectory, files)