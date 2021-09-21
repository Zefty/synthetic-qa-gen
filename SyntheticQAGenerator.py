import argparse
import csv
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
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
    
    def generate_answer(self, passage, use_pipeline = True):
        if use_pipeline: 
            return self.answer_summarizer_pipeline(passage, truncation = True)[0]['summary_text']
        else: 
            inputs = self.as_tokenizer.encode("summarize: " + passage, return_tensors = "pt", max_length = 512, truncation = True).to(0)
            outputs = self.as_model.generate(inputs, length_penalty = 2.0, num_beams = 4, early_stopping = True)
            return self.as_tokenizer.decode(outputs[0]).replace('</s>', '')

    def generate_answers(self, passages, use_pipeline = True):
        summaries = []
        if use_pipeline:
            summaries.extend(self.answer_summarizer_pipeline(passages, truncation = True))
            return summaries
        else: 
            for passage in passages:
                inputs = self.as_tokenizer.encode("summarize: " + passage, return_tensors = "pt", max_length = 512, truncation = True).to(0)
                outputs = self.as_model.generate(inputs, length_penalty = 2.0, num_beams = 4, early_stopping = True)
                summaries.append(self.as_tokenizer.decode(outputs[0]))
            return summaries

    def generate_question(self, passage, answer):
        qg_input = "{} {} {} {}".format(
            self.ANSWER_TOKEN, answer, self.CONTEXT_TOKEN, passage
        )
        self.qg_model.eval()
        encoded_input = self.qg_tokenizer(qg_input, padding = 'max_length', max_length = self.SEQ_LENGTH, truncation = True, return_tensors = "pt").to(self.device)
        with torch.no_grad():
            output = self.qg_model.generate(input_ids = encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(output[0], skip_special_tokens = True)
        return question[0:question.find('?') + 1]

    def make_dict(self, question, answer):
        qa = {}
        qa["question"] = question
        qa["answer"] = answer
        return qa

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

    with open(args.csv_path) as csv_file:
        no_of_lines = sum(1 for row in csv_file)

    with open(args.csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')

        for index, row in tqdm(enumerate(csv_reader), total = no_of_lines):

            answer = qa_gen.generate_answer(row[1], use_pipeline = False)
            question = qa_gen.generate_question(row[1], answer)

            if answer == '' or question == '':
                continue 

            split = random.uniform(0, 1)
            if split > 0.1:
                train_source.write(question + '\n')
                train_target.write(answer + '\n')
            else:
                val_source.write(question + '\n')
                val_target.write(answer + '\n')

if __name__ == "__main__":
    main()