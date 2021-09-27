import csv
import os
import json
import re

def CreateRAGTrainingData(indirectory, outdirectory, files):
    for file in files:
        with open('{}/{}'.format(outdirectory, file), 'w', encoding = 'utf8') as outfile:
            with open('{}/{}'.format(indirectory, file), 'r', encoding = 'utf8') as infile:
                    for i in range(100000):
                        outfile.write(infile.readline())

def CreateRAGTrainingData2(indirectories, num_rows, outdirectory, files):
    for file in files:
        with open('{}/{}'.format(outdirectory, file), 'w', encoding = 'utf8') as outfile:
            for index, indirectory in enumerate(indirectories):
                with open('{}/{}'.format(indirectory, file), 'r', encoding = 'utf8') as infile:
                        for i in range(num_rows[index]):
                            outfile.write(infile.readline())

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