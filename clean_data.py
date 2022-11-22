import json
import pandas as pd
import time
import re
from underthesea import text_normalize

from tqdm import tqdm

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
    "]+", flags=re.UNICODE)
def remove_emoji(text):
    return emoji_pattern.sub(r' ', text)

PUNCTUATIONS = ["''", "'", "``", "`", "-", "--"] 
def remove_punctuations(text):
    for p in PUNCTUATIONS:
        text = text.replace(p, ' ')
    return text

ICONS = [':)))))', ':))))', ':)))', ':))', ':)', ':D', ':v' \
    ':(((((', ':((((', ':(((', ':((', ':(' \
        '(y)', ':">', ':\'(', ':|', ':-)' 
]
def remove_icons(text):
    for icon in ICONS:
        text = text.replace(icon, ' ')
    return text


def preprocess(text):
    text = remove_emoji(text)
    text = remove_punctuations(text)
    text = remove_icons(text)
    text = text_normalize(text)
    return text

def main(root, tar):
    with open(root, 'r') as f:
        data = json.load(f)
        
    for item in tqdm(data):
        rev_id = item['RevId']
        comment = item['Comment']
        comment = preprocess(comment)
        item['Comment'] = comment
        rating = item['Rating']
        sentiment = int(rating >= 6.0)
        item['Sentiment'] = sentiment

        # images = []
        # for image_url in item['image_urls']:
        #     image_url = image_url.split('/')[-1]
        #     images.append(image_url)
        # item['images'] = images

    df = pd.DataFrame(data)
    del df['image_urls']
    df.to_csv(tar, index=False)

if __name__ == '__main__':
    root = 'data/train.json'
    tar = 'data/train_clean.csv'
    main(root, tar)



