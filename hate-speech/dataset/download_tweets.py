# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 09:44:42 2018

@author: aitor
"""

import csv, json, sys, time

import tweepy

DATASET= "NAACL_SRW_2016.csv"
DATASET_COMPLETE = "dataset_complete.csv"
CONSUMER_KEY = 'cepLiWNRET3UYxBZJb5v7fp9s'
CONSUMER_SECRET = 'CLCfDEcEyvk3hVaIwwAHaasVuAKc0N9RYWktHuZV2aCtHhgz8Q'
OAUTH_TOKEN = '419532171-ruHDFCE419sTeLxEAWfDNQwmNSctSlvDYqfiCvPA'
OAUTH_TOKEN_SECRET = 'l0IlJhq97feWfqbe15U4t52dTAaSt880lZtWiqgAo3ZlN'

    
if __name__ == "__main__":
    print 'Starting'
    print 'Creating connection...'
    sys.stdout.flush()  
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)
    print 'Dowloading tweets...'
    sys.stdout.flush()  
    rows = []
    with open(DATASET, 'r') as csv_file:
        reader = csv.reader(csv_file)  
        i = 0
        for row in reader:
            i += 1
            try:
                tweet_id = row[0]
                tweet = api.get_status(tweet_id)   
                text = tweet.text
                label = row[1]
                rows.append([text, label])
            except:
                print 'Error downloading tweet', tweet_id
                sys.stdout.flush()  
            print 'Downloaded', i
            sys.stdout.flush()  
            time.sleep(5)
    print 'Training examples:', len(rows)
    sys.stdout.flush()  

    with open(DATASET_COMPLETE, 'w') as clean:
        json.dump(rows, clean, indent = 2)

    print 'end'

