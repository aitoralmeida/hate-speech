# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:39:12 2018

@author: aitor
"""

import json, sys
from unidecode import unidecode

DATASET_IRONY = "./../irony/dataset/irony-labeled-clean.json"
DATASET_VALENCE_AROUSAL = "./../valence_arousal/dataset/valence-arousal-clean.json"
DATASET_POLARITY = "./../polarity/dataset/full-corpus-clean.json"
DATASET_EMOTION = "./../emotion/dataset/text_emotion-clean.json"
DATASET_HATE_DT = "./../hate-speech/dataset/dataset_clean_DT.json"
DATASET_HATE_RM = "./../hate-speech/dataset/dataset_clean_RM.json"
DATASET_HATE_WZ_L = "./../hate-speech/dataset/dataset_clean_WZL.json"
DATASET_SO = "./../stackoverflow/dataset/stack_comments_clean.json"


datasets_json = [DATASET_IRONY, DATASET_VALENCE_AROUSAL, DATASET_POLARITY, DATASET_EMOTION, DATASET_SO]#, DATASET_HATE_DT, DATASET_HATE_RM,DATASET_HATE_WZ_L]

unique_words = set()
max_phrase_length = 0

  
if __name__ == "__main__":
    print 'Processing datasets'
    sys.stdout.flush()
    with open('total_text.txt', 'w') as outfile:
        outfile.write(' ')        
    total_text = ''
    i = 0
    for dataset_json in datasets_json:
        print 'Dataset:', dataset_json 
        sys.stdout.flush()
        with open(dataset_json, 'r') as dataset_file:
            rows = json.load(dataset_file)
        i = 0
        total = len(rows)
        accum_text = ' '
        for row in rows:
            i += 1
            text = row[0]
            accum_text = accum_text + ' ' + text
            words = set(text.split(' '))
            unique_words = unique_words.union(words)
            if len(words) > max_phrase_length:
                max_phrase_length = len(words)
            if i % 5000 == 0:
                print 'Processed line', i, 'of', total
                with open('total_text.txt', 'a') as outfile:
                    print len(accum_text)
                    print 'Writing File'
                    outfile.write(unidecode(accum_text))
                    accum_text = ' '
            sys.stdout.flush()
        with open('total_text.txt', 'a') as outfile:
            print len(accum_text)
            print 'Writing File'
            outfile.write(unidecode(accum_text))
            accum_text = ' '
            sys.stdout.flush()
                
                    
    print len(unique_words)
    print max_phrase_length  
    sys.stdout.flush()
    with open('unique_words.json', 'w') as outfile:
        json.dump(list(unique_words), outfile)
    with open('max_phrase_length.json', 'w') as outfile:
        json.dump(max_phrase_length, outfile)
#    with open('total_text.txt', 'w') as outfile:
#        outfile.write(unidecode(total_text))
    print 'FIN'              
    
                
    