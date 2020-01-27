# VADER Sentiment Analysis.
# 
# This script calculates the text review sentiment from the yelp.csv dataset and exports the scores
# into vader_sentence_sentiment.csv
#
#
# VADER (Valence Aware Dictionary and sEntiment Reasoner)
# is a lexicon and rule-based sentiment analysis tool that is specifically attuned
# to sentiments expressed in social media, and works well on texts from other domains. 
# See: https://github.com/cjhutto/vaderSentiment
#
# For each review text, we extracted the sentiment scores as an average across
# sentences('sentence average').
# We also output sentiment score for the sentince with highest absolute value of compound
# sentiment ('strongest sentiment statement')
#
# Outputs:
#    'neg_avg': negative sentiment (sentence average)', 
#    'neu_avg': neutral sentiment (sentence average)', 
#    'pos_avg': positive sentiment (sentence average)', 
#    'comp_avg': composite sentiment (sentence average)', 
    
#    'neg_max': negative sentiment (strongest sentiment statement)',  
#    'neu_max': neutral sentiment (strongest sentiment statement)', 
#    'pos_max': negative sentiment (strongest sentiment statement)', 
#    'comp_max': composite sentiment (strongest sentiment statement)', 
        
        
# Import dependencies
from statistics import mean
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize
import numpy as np
from tqdm import tqdm
import collections

# load data
review = pd.read_csv('C:/Users/adamb/Desktop/vader/yelp_review.csv')

tqdm.pandas() # progress bar
analyser = SentimentIntensityAnalyzer() # create analyser

# Function to get scores
def sentiment_analyzer_scores(text):
    """Calculates sentiment scores for each text review
    
    Parameters
    ----------
    text : str
        Text to analyze

    Returns
    ----------
    list of float:
    [0] 'neg_avg': negative sentiment (sentence average)', 
    [1] 'neu_avg': neutral sentiment (sentence average)', 
    [2] 'pos_avg': positive sentiment (sentence average)', 
    [3] 'comp_avg': composite sentiment (sentence average)', 
    [4] 'neg_max': negative sentiment (strongest sentiment statement)',  
    [5] 'neu_max': neutral sentiment (strongest sentiment statement)', 
    [6] 'pos_max': negative sentiment (strongest sentiment statement)', 
    [7] 'comp_max': composite sentiment (strongest sentiment statement)', 
    """
    
    score_list=[]
    sent_text = sent_tokenize(text) # list of sentences
    for sentence in sent_text: # for each sentece
        score = analyser.polarity_scores(sentence) # measure sentiment for each sentence
        score_list.append(score)
    
    # sentence with strongest sentiment
    max_s = max(score_list, key=lambda x:abs(x['compound'])) 
    
    # sentiment averaged across sentences
    mean_s = {}
    for key in score_list[0].keys():
        mean_s[key] = sum(d[key] for d in score_list) / len(score_list)
  
    # list of scores
    score=[mean_s['neg'], mean_s['neu'],mean_s['pos'],mean_s['compound'],
       max_s['neg'], max_s['neu'],max_s['pos'],max_s['compound']]
    
    return score


# Analyse all reviews
print('Running')
sentiment =review['text'].progress_apply(sentiment_analyzer_scores)
sentiment_df=sentiment.apply(pd.Series)
sentiment_df.columns = ['neg_avg', 'neu_avg', 'pos_avg' ,'comp_avg','neg_max', 'neu_max', 'pos_max' ,'comp_max']

# Save scores to csv
sentiment_df.to_csv('C:/Users/adamb/Desktop/vader/vader_sentence_sentiment.csv')

print('Done')







