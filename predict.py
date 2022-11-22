import numpy as np
import pandas as pd
import yaml

import os

from tqdm import tqdm
from eval import eval_metrics

import argparse

import underthesea
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
import py_vncorenlp

def underthesea_predict(texts):
    try:
        print('Using underthesea setiment analysis')
    except:
        print('Please install underthesea')
        return None
    sentiments = []
    for text in tqdm(texts):
        try:
            sentiment = int(underthesea.sentiment(text) == 'positive')
        except:
            sentiment = 1
        sentiments.append(sentiment)

    return sentiments

def phobert_classify_pretrain_predict(config, texts):
    print('Using pretrained phobert setiment analysis')
    print('Model: {}'.format(config['model']))
    print('Tokenizer: {}'.format(config['tokenizer']))

    model = RobertaForSequenceClassification.from_pretrained(config['model'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=config['rdrsegmenter'])

    sentiments = []
    for text in tqdm(texts): 
        try:
            text = rdrsegmenter.word_segment(text)
            text = ' '.join(text)
            input_ids = torch.tensor([tokenizer.encode(text, truncation=True, max_length=256)])

            with torch.no_grad():
                out = model(input_ids)
                out = out.logits.softmax(dim=-1).tolist()
            if out[0][0] > out[0][1]:
                sentiment = 0
            else:
                sentiment = 1
            sentiments.append(sentiment)
            # sentiments.append(1)
        except:
            sentiments.append(1)

    return sentiments
        
def write_metrics(vote, metrics, metric_path):
    with open(metric_path, 'w') as f:
        f.write('Vote: {}\n'.format(vote))
        f.write("accuracy: " + str(metrics[0]) + "\n")
        f.write("precision: " + str(metrics[1]) + "\n")
        f.write("recall: " + str(metrics[2]) + "\n")
        f.write("f1: " + str(metrics[3]) + "\n")
        f.write("auc: " + str(metrics[4]) + "\n")

def main(config, df, output_path, metric_path):
    vote_list = []
    if config['underthesea']['enable']:
        sentiments = underthesea_predict(df['Comment'])
        vote_list.append('underthesea')
        df['underthesea'] = sentiments
    if config['phobert_pretrain']['enable']:
        sentiments = phobert_classify_pretrain_predict(config['phobert_pretrain'], df['Comment'])
        vote_list.append('phobert_pretrain')
        df['phobert_pretrain'] = sentiments

    df.to_csv(output_path, index=False, encoding='utf-8')

    y_true = df['Sentiment']

    for vote in vote_list:
        y_pred = df[vote]
        metrics = eval_metrics(y_true, y_pred)
        print('Vote: {}'.format(vote))
        print(metrics)

        write_metrics(vote, metrics, metric_path)

    vote_sentiment = []
    for i in range(len(df)):
        vote = 0
        for col in vote_list:
            vote += df[col][i]
        vote_sentiment.append(int(vote >= len(vote_list) / 2))
    df['vote'] = vote_sentiment
    metrics = eval_metrics(y_true, df['vote'])
    print('Vote: {}'.format('vote'))
    print(metrics)

    write_metrics('vote', metrics, metric_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to input file', default='data/train_clean.csv')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.yaml')
    parser.add_argument('--output_path', type=str, help='Path to output file', default='train_sentiment.csv')
    parser.add_argument('--metric_path', type=str, help='Path to metric file', default='metrics.txt')
    args = parser.parse_args()

    # load the test data
    input_path = args.input_path
    df = pd.read_csv(input_path)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config, df, args.output_path, args.metric_path)

