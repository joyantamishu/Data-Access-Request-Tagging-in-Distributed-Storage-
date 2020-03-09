#I used and modify the code upload here: https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np


df = pd.read_csv('process_info_flowlet.csv',delimiter=',',encoding='latin-1')

unique_element_list = df['process'].unique()

unique_list_map = dict()

count = 0
for single_element in unique_element_list:
    unique_list_map[single_element] = count
    count = count + 1

print(unique_list_map)

df['process'] = df.process.map(unique_list_map)

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['access'])

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts, df['process'], test_size=0.2)

model = MultinomialNB().fit(X_train, y_train)

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))


#print(df['process'])