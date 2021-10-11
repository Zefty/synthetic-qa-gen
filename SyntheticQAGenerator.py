import argparse
import csv
import math
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import sentencepiece

class SyntheticQAGenerator:
    '''Unambiguous synthetic question-answer generator via abstractive summarisation.
    This class uses a SCI-TLDR model to summarise passages to be used as a synthetic answer.
    This class uses a T5 model to generate a synthetic question.  
    '''

    def __init__(self):
        '''initialise class 
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SCI-TLDR model for answer generation via passage summarisation
        AS_PRETRAINED = 'lrakotoson/scitldr-catts-xsum-ao'
        self.as_tokenizer = AutoTokenizer.from_pretrained(AS_PRETRAINED)
        self.as_model = AutoModelForSeq2SeqLM.from_pretrained(AS_PRETRAINED)
        self.as_model.to(self.device)
        self.answer_summarizer_pipeline = pipeline("summarization", model = self.as_model, tokenizer = self.as_tokenizer, device = 0) # Device = 0 allows us to use the GPU

        # T5 model for question generation 
        QG_PRETRAINED = 'iarfmoose/t5-base-question-generator'
        self.ANSWER_TOKEN = '<answer>'
        self.CONTEXT_TOKEN = '<context>'
        self.SEQ_LENGTH = 1024
        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.question_generator_pipeline = pipeline("text2text-generation", model = self.qg_model, tokenizer = self.qg_tokenizer, device = 0)
    
    def generate_answers(self, passages):
        '''generate_answers takes a list of passages and generates a summary for each passage
        :param passages: [list] 
        :output answers: [list]
        '''
        summaries = self.answer_summarizer_pipeline(['summarize: ' + passage for passage in passages], truncation = True)
        return [answer['summary_text'] for answer in summaries]
    
    def generate_questions(self, passages, answers):
        '''generate_questions takes a list of passages and a list of answers to generate a list of questions
        :param passages: [list] 
        :param answers: [list] 
        :output questions: [list]
        '''
        inputs = ['{} {} {} {}'.format(
            self.ANSWER_TOKEN, pair[1], self.CONTEXT_TOKEN, pair[0]
        ) for pair in zip(passages, answers)]
        questions = self.question_generator_pipeline(inputs, truncation = True)
        return [question['generated_text'][0:question['generated_text'].find('?') + 1] for question in questions]

def generate(data, split, chunk_size = 5):
    '''generate takes a list of passages and generates a list of synthetic QA pairs
    :param data: [list] 
    :param split: str
    :param chunk_size: int
    '''
    # Create synthetic qa generator object from class 
    # The class contains a SCI-TLDR model for answer sumarisation
    # The class contains a T5 model for question generation 
    qa_gen = SyntheticQAGenerator()

    # Open files for saving QA pairs
    source = open('{}.source'.format(split), 'w')
    target = open('{}.target'.format(split), 'w')

    # Process passages in parallel given chunk_size 
    numberChunks = math.ceil(len(data) / chunk_size) 
    for i in tqdm(range(numberChunks), position = 0, leave = True):
        passages = data['text'][i*chunk_size:(i + 1)*chunk_size]

        # Generate synthetic QA pair
        answers = qa_gen.generate_answers(passages)
        questions = qa_gen.generate_questions(passages, answers)
        
        # Remove empty results
        for j in zip(answers, questions):
            if j[0] == '' or j[1] == '':
                continue 

            # Save synthetic QA pairs
            source.write(j[1] + '\n')
            target.write(j[0] + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default='splitted_covid_dump-covidQA.csv',
        type=str,
        required=True,
        help="Path to a tab-separated csv file with columns 'title' and 'text'.",
    )
    parser.add_argument(
        "--shard",
        default=False,
        type=bool,
        required=False,
        help="Whether to process the csv file in chunks/shards.",
    )
    parser.add_argument(
        "--start_shard",
        default=0,
        type=int,
        required=False,
        help="Starting index for chunk/shard of the csv file.",
    )
    parser.add_argument(
        "--end_shard",
        default=20,
        type=int,
        required=False,
        help="Ending index for chunk/shard of the csv file.",
    )
    parser.add_argument(
        "--num_shards",
        default=20,
        type=int,
        required=False,
        help="Number of chunks/shards to process for the csv file.",
    )
    parser.add_argument(
        "--chunk_size",
        default=10,
        type=int,
        required=False,
        help="Number of passages to process concurrently. For generating question/answer pairs.",
    )
    args = parser.parse_args()
    
    covid_dump = load_dataset('csv', data_files = args.csv_path, delimiter = '\t', column_names = ['title', 'text'])
    covid_dump = covid_dump['train'].train_test_split(test_size = 0.1, shuffle = False, seed = 42)

    print('[Generating Synthetic QA Data]')
    if args.shard:
        for i in range(args.start_shard, args.num_shards):
            print('[Shard: {}]'.format(i))
            if i == args.end_shard:
                break
            generate(covid_dump['train'].shard(num_shards = args.num_shards, index = i), split = 'train_chunk_{}'.format(i), chunk_size = args.chunk_size)
            generate(covid_dump['test'].shard(num_shards = args.num_shards, index = i), split = 'val_chunk_{}'.format(i), chunk_size = args.chunk_size)
    else:
        print('[Full Generation]')
        generate(covid_dump['train'], split = 'train')
        generate(covid_dump['test'], split = 'val')

if __name__ == "__main__":
    main()
