import argparse
import csv
import math
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import sentencepiece

import random
random.seed(42)

class SyntheticQAGenerator:
    def __init__(self, model_dir = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        AS_PRETRAINED = 'lrakotoson/scitldr-catts-xsum-ao'
        self.as_tokenizer = AutoTokenizer.from_pretrained(AS_PRETRAINED)
        self.as_model = AutoModelForSeq2SeqLM.from_pretrained(AS_PRETRAINED)
        self.as_model.to(self.device)
        self.answer_summarizer_pipeline = pipeline("summarization", model = self.as_model, tokenizer = self.as_tokenizer, device = 0) # Device = 0 allows us to use the GPU

        QG_PRETRAINED = 'iarfmoose/t5-base-question-generator'
        self.ANSWER_TOKEN = '<answer>'
        self.CONTEXT_TOKEN = '<context>'
        self.SEQ_LENGTH = 1024
        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.question_generator_pipeline = pipeline("text2text-generation", model = self.qg_model, tokenizer = self.qg_tokenizer, device = 0)

    def generate_answers(self, passages):
        summaries = self.answer_summarizer_pipeline(['summarize: ' + passage for passage in passages], truncation = True)
        return [answer['summary_text'] for answer in summaries]
    
    def generate_questions(self, passages, answers):
        inputs = ["{} {} {} {}".format(
            qa_gen.ANSWER_TOKEN, pair[1], qa_gen.CONTEXT_TOKEN, pair[0]
        ) for pair in zip(passages, answers)]
        questions = self.question_generator_pipeline(inputs, truncation = True)
        return [question['generated_text'][0:question['generated_text'].find('?') + 1] for question in questions]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default='splitted_covid_dump-covidQA.csv',
        type=str,
        required=True,
        help="Path to a tab-separated csv file with columns 'title' and 'text'.",
    )
    args = parser.parse_args()

    qa_gen = SyntheticQAGenerator()

    train_source = open('train.source', 'w')
    train_target = open('train.target', 'w')
    val_source = open('val.source', 'w')
    val_target = open('val.target', 'w')

    covid_dump = load_dataset('csv', data_files = args.csv_path, delimiter = '\t', column_names = ['title', 'text'])
    chunkSize = 10
    numberChunks = math.ceil(len(covid_dump['train']) / chunkSize) 
    for i in tqdm(range(numberChunks), position = 0, leave = True):
        passages = covid_dump['train']['text'][i*chunkSize:(i + 1)*chunkSize]
        answers = qa_gen.generate_answers(passages)
        questions = qa_gen.generate_questions(passages, answers)
        
        for j in zip(answers, questions):
            if j[0] == '' or j[1] == '':
                continue 
            
            split = random.uniform(0, 1)
            if split > 0.1:
                train_source.write(j[1] + '\n')
                train_target.write(j[0] + '\n')
            else:
                val_source.write(j[1] + '\n')
                val_target.write(j[0] + '\n')

if __name__ == "__main__":
    main()