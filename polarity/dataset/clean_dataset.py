# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:03:24 2018

@author: aitor
"""

import csv, json

DATASET= "full-corpus.csv"
DATASET_CLEAN = "full-corpus-clean.json"


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
        reader = csv.reader(csv_file)         
        for row in reader:
            text = row[4]
            text = clean_text(text)
            label = row[1]
            rows.append([text, label])

    with open(DATASET_CLEAN, 'w') as clean:
        json.dump(rows, clean, indent = 2)
    
#    with open(DATASET_IRONY_CLEAN, 'r') as intermediate:
#        for line in intermediate.readline():
#            print '-----' + line
##        with open('irony-labeled-clean.csv', 'w') as target:
##            print intermediate.read()
    
    print 'end'
