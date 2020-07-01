# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:41:28 2020

@author: vai22
"""
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import os

from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'Data')
stop_words = open('vietnamese_stopwords.txt',encoding="utf8")
a=stop_words.read()
my_stopwords = a
# print(my_stopwords)

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = lines.replace('\ n', ' ')
                #Xử lí các kí tự đặc biệt
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                #Tách từ
                lines = ViTokenizer.tokenize(lines)         
                X.append(lines)
                y.append(path)
    return X, y


#Hàm ghi file
def writeFile(txtArrAfter, outputName, path):
    f = open(path + "/" + outputName + ".txt", 'a',encoding="utf-8")
    f.write(str(txtArrAfter))
    f.close()

#Load dữ liệu vào X_data, y_data
train_path = os.path.join(dir_path, 'C:/Users/vai22/OneDrive/Desktop/Data/Train_Full')
X_data, y_data = get_data(train_path)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)

# transform the training and validation data using count vectorizer object
X_train_count = count_vect.transform(X_data)



# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)
# assume that we don't have test set before


# writeFile(X_data,'X_data', 'C:/Users/vai22/OneDrive/Desktop/final/Data/vnexpress.net')
# writeFile(y_data,'y_data', 'C:/Users/vai22/OneDrive/Desktop/final/Data/vnexpress.net')



X_train, X_test, y_train, y_test = train_test_split(X_data_tfidf, y_data,random_state = 0, test_size=0.3)

print("Training size: ", len(y_train))
print("Test size    : ", len(y_test))


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print(y_predict)
print(y_test)

print("Accuracy of kNN: ", (100 * accuracy_score(y_test, y_predict)),"%")
#build nearest matrix
neigh = NearestNeighbors(n_neighbors = 5)
neigh.fit(X_train)
#looking for some nearest
(distance, found_index) = neigh.kneighbors(X_test)
#tính độ chính xác 
accuracy = knn.score(X_test,y_test)
print("Độ chính xác của kNN:"+str(accuracy))
#precision recall f1
print("Precision:",precision_score(y_test,y_predict, average = None))
print("Recall:",recall_score(y_test,y_predict, average = None))
print("F1-Score:",f1_score(y_test,y_predict, average = None))



                   
#

