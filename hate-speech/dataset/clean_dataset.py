# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:03:24 2018

@author: aitor
"""

import csv, json

DATASET_DT= "labeled_data.csv"
DATASET_CLEAN_DT = "dataset_clean_DT.json"
DATASET_RM= "labeled_data_tweets_only.csv"
DATASET_CLEAN_RM = "dataset_clean_RM.json"


'''


ID, count,hate_speech,offensive_language,neither,class,tweet

ID

count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

hate_speech = number of CF users who judged the tweet to be hate speech.

offensive_language = number of CF users who judged the tweet to be offensive.

neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.

class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
'''


def clean_text(text):
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace(';', '')
    text = text.replace(':', '')
    text = text.replace('?', '')
    text = text.replace('¿', '')
    text = text.replace('¡', '')
    text = text.replace('!', '')
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace('--', ' ')
    text = text.replace('#', '')
    text = text.replace('@', '')
    text = text.replace('%', '')
    text = text.replace('&', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('=', '')
    text = text.replace('$', '')
    text = text.replace('€', '')
    text = text.replace('+', '')
    text = text.replace('-', '')
    text = text.replace('*', '')
    text = text.replace('~', '')
    text = text.replace('`', '') 
    text = text.replace('|', '') 
    text = text.replace('--', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('/', ' ')  
    text = text.replace('=', ' ') 
    text = text.replace('>', ' ') 
    text = text.replace('<', ' ') 
    text = text.replace('\\', ' ') 
    text = text.replace('^', ' ') 
    text = text.replace('_', ' ')     
    text = text.strip()
    text = text.lower()
    return text
    
if __name__ == "__main__":
    print 'starting'
   
    print 'DT'
    rows = []
    with open(DATASET_DT, 'r') as csv_file:
        reader = csv.reader(csv_file)         
        for row in reader:
            text = row[6]
            text = clean_text(text)
            label = row[5]
            rows.append([text, label])
    print 'Training examples:', len(rows)

    with open(DATASET_CLEAN_DT, 'w') as clean:
        json.dump(rows, clean, indent = 2)
        
    print 'RM'    
    rows = []
    with open(DATASET_RM, 'r') as csv_file:
        reader = csv.reader(csv_file)         
        for row in reader:
            text = row[1]
            text = clean_text(text)
            label = row[0]
            rows.append([text, label])
    print 'Training examples:', len(rows)

    with open(DATASET_CLEAN_RM, 'w') as clean:
        json.dump(rows, clean, indent = 2)
    
#    with open(DATASET_IRONY_CLEAN, 'r') as intermediate:
#        for line in intermediate.readline():
#            print '-----' + line
##        with open('irony-labeled-clean.csv', 'w') as target:
##            print intermediate.read()
    print 'end'
