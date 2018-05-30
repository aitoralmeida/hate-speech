# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:03:24 2018

@author: aitor
"""

import csv, json

DATASET= "dataset-fb-valence-arousal-anon.csv"
DATASET_CLEAN = "valence-arousal-clean.json"


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
    rows = []
    with open(DATASET, 'r') as csv_file:
        for line in csv_file:
            tokens = line.split(',')
            valence1 = float(tokens[-1])
            valence2 = float(tokens[-2])
            valence = (valence1 + valence2) / 2.0
            valence = valence / 10.0
            arousal1 = float(tokens[-3])
            arousal2 = float(tokens[-4])
            arousal = (arousal1 + arousal2) / 2.0
            arousal = arousal / 10.0
            if len(tokens) > 5:
                print line
                text = ''
                for i in range(len(tokens)-4):
                    text += tokens[i]
            else:
                text = tokens[0]
            text = clean_text(text)
            rows.append([text, valence, arousal])

    with open(DATASET_CLEAN, 'w') as clean:
        json.dump(rows, clean, indent = 2)
    
#    with open(DATASET_IRONY_CLEAN, 'r') as intermediate:
#        for line in intermediate.readline():
#            print '-----' + line
##        with open('irony-labeled-clean.csv', 'w') as target:
##            print intermediate.read()
    
    print 'end'
