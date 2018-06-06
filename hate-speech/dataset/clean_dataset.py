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
DATASET_WZ_L_RACISM = "./hs_data/racism.json"
DATASET_WZ_L_SEXISM = "./hs_data/sexism.json"
DATASET_WZ_L_NEITHER = "./hs_data/neither.json"
DATASET_WZ_L = "dataset_clean_WZL.json"


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
            if label == '0' or label == '2':
                rows.append([text, label])
            else:
                print label
    print 'Training examples:', len(rows)    
    
    with open(DATASET_CLEAN_RM, 'w') as clean:
        json.dump(rows, clean, indent = 2)
        
    print 'WZ_L'
    rows = []    
    #I'm not proud of this
    with open(DATASET_WZ_L_RACISM, 'r') as j_file:
        for line in j_file:
            tokens = line.split('"truncated":')
            tokens = tokens[1].split('"text":')
            tokens = tokens[1].split(', "created_at":')
            text = tokens[0]
            if ', "extended_entities"' in tokens[0]:
                text = text.split(', "extended_entities"')[0]
            if ', "place":' in tokens[0]:
                text = text.split(', "place":')[0]  
            if ', "is_quote_status"' in tokens[0]:
                text = text.split(', "is_quote_status"')[0] 
            text = clean_text(text)
            label = '1'
            rows.append([text, label])
            
    with open(DATASET_WZ_L_SEXISM, 'r') as j_file:
        for line in j_file:
            tokens = line.split('"truncated":')
            tokens = tokens[1].split('"text":')
            tokens = tokens[1].split(', "created_at":')
            text = tokens[0]
            if ', "extended_entities"' in tokens[0]:
                text = text.split(', "extended_entities"')[0]
            if ', "place":' in tokens[0]:
                text = text.split(', "place":')[0]   
            if ', "is_quote_status"' in tokens[0]:
                text = text.split(', "is_quote_status"')[0] 
            text = clean_text(text)
            label = '2'
            rows.append([text, label])
            
    with open(DATASET_WZ_L_NEITHER, 'r') as j_file:
        for line in j_file:
            tokens = line.split('"text":')
            tokens = tokens[1].split(', "coordinates":')
            text = tokens[0]
            if ', "extended_entities"' in tokens[0]:
                text = text.split(', "extended_entities"')[0]
            if ', "place":' in tokens[0]:
                text = text.split(', "place":')[0]   
            if ', "is_quote_status"' in tokens[0]:
                text = text.split(', "is_quote_status"')[0] 
            text = clean_text(text)
            label = '0'
            rows.append([text, label])           
            
    print 'Training examples:', len(rows)    
    
    with open(DATASET_WZ_L, 'w') as clean:
        json.dump(rows, clean, indent = 2)
    
#    with open(DATASET_IRONY_CLEAN, 'r') as intermediate:
#        for line in intermediate.readline():
#            print '-----' + line
##        with open('irony-labeled-clean.csv', 'w') as target:
##            print intermediate.read()
    print 'end'
