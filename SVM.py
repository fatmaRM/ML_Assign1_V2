#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import json    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score


#import file that read data from json file 
from read_data import read_clean_data

#CALL read_clean_data function to have all cleaned articles along with all nummirc ids mapped to categories 
articles_df , category_to_id  = read_clean_data();

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word' ,sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#body is transformed and mapped to features 
features = tfidf.fit_transform(articles_df.body).toarray()
# classes labels 
labels = articles_df.cat_id
#SVM Model 
model = LinearSVC()
training_articles, testing_articles, training_labeles, testing_labels , indices_train, indices_test = train_test_split(features, labels, articles_df.index, test_size=0.33, random_state=0)
model.fit(training_articles, training_labeles)
predicted_labels = model.predict(testing_articles)

print("SVM accuracy ",accuracy_score(testing_labels,predicted_labels))


# In[16]:


#from sklearn.metrics import confusion_matrix
#conf_mat = confusion_matrix(y_test, y_pred)
#fig, ax = plt.subplots(figsize=(10,10))
#sns.heatmap(conf_mat, annot=True, fmt='d',
 #           xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




