# Yelp Case Study
## Predicting Businesses Closure

***Aim:***   
The goal was to build a classifier that would predict whether business will close or not based on the Yelp dataset (https://www.kaggle.com/yelp-dataset/yelp-dataset/version/6). The motivation was to create a model that would give investors a chance to better asses the risk of business failure and to provide them with information that would help them to make an informed decision whether they should invest in given business or not.

___

***Requirments:***   
Implemented in Python 3.7 using the following packages:   
*anaconda, seaborn, mpl_toolkits, vaderSentiment, sklearn, imblearn* 

___
***The projects contains following scripts:***

***Build_dataset.ipynb*** performs Yelp data import, data visualization/exploration, feature extraction and preparation for classification. Exports features and their descriptions.   

*Input: 'yelp_business.csv', 'yelp_business_attributes.csv', 'yelp_review.csv'*   
*Output: feat_mat_clean.csv',  'feature_description.csv'*   

***vader_sentence.py*** calculates the text review sentiment from the yelp.csv dataset and exports the sentiment scores.   

*Input: 'review.csv'*   
*Output: 'vader_sentence_sentiment.csv'*   

***Run_Classification.ipnb*** implements the classification using several different algorithms.  

*Input: 'feat_mat_clean.csv', 'feature_description.csv'*    
