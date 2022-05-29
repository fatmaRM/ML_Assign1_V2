#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import json    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def read_clean_data():
    #Python dictionary train_data to hold parsed json file 
    train_data = []
    #read json file into datframe
    with open ('articles.json') as f_train : train_data = json.load(f_train)
    #dataframe holds all article 
    articles_df = pd.DataFrame (train_data, columns = ['body','title','category'])  
    ###########clean data 
    # check if there is a duplicate body 
    from collections import Counter
    counter_similar_bodies = Counter(articles_df['body'])
    counter_similar_bodies.most_common(1)
    orginal_length = (len(articles_df['body']))
    duplicated_length = len(counter_similar_bodies)
   # if (duplicated_length<orginal_length): print ('duplicate exit')
    #remove duplicate row that contains duplicate body 
    #keep first 
    articles_df.drop_duplicates(subset ="body",keep = 'first', inplace = True)    
   # print(pd.unique(articles_df['category']))#3 CLASS  ['Engineering', 'Startups & Business', 'Product & Design']
    articles_df['cat_id']=articles_df['category'].factorize()[0]; #map category to id from 0 to 2 
    # category to numeric id  mapping 
    category_id_df = articles_df[['category', 'cat_id']].drop_duplicates().sort_values('cat_id')
    id_to_category = (category_id_df[['cat_id', 'category']].values)
    category_to_id = dict(category_id_df.values)    
    return articles_df,category_to_id;

def plot_read_data():
    articles_df , category_to_id  = read_clean_data();
    #############data ploting and insight 
    #plot data to check if it is imbalanced or not 
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    articles_df.groupby('category').body.count().plot.bar(ylim=0)
    print ("view details about data body for 2 n-grams")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(articles_df.body).toarray()
    labels = articles_df.cat_id
    print (features.shape)
    from sklearn.feature_selection import chi2
    import numpy as np
    N=2
    for category, cat_id in sorted(category_to_id.items()):
      features_chi2 = chi2(features, labels == cat_id)
      indices = np.argsort(features_chi2[0])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
      print("# '{}':".format(category))
      print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
      print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))



# In[ ]:




