'''
Quora classific HR
'''

# Enter your code here. Read input from STDIN. Print output to STDOUT

import os
import string
import json
import numpy as np 
import sklearn
import scipy

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack

data_dir_inp = '/Users/denisaroberts/Desktop/twosig/craig/craigslist-post-classifier-the-category-testcases/input/'
data_dir_out = '/Users/denisaroberts/Desktop/twosig/craig/craigslist-post-classifier-the-category-testcases/output/'

def load_data_inp(filename):
	'''parse first int line and then json objects, get feature vectors. Will have to preprocess X after'''
	X = []

	with open(os.path.join(data_dir_inp, filename), 'r') as f:
		N = int(f.readline())
		
		for i in range(N):
			line = f.readline()
			line = json.loads(line)
			temp = list(line.values())[2].strip().split(' ')
			
			temp = [preprocess(word) for word in temp]
			# remove non-words
            # tfidf removes stop words direclty
			temp = [w for w in temp if w.isalpha()]
			
			temp1 = list(line.values())[:2]
			temp1.append(' '.join(temp))
			X.append(temp1)
					
	return np.array(X)

def load_data_out(filename, N):
	y = []

	with open(os.path.join(data_dir_out, filename), 'r') as f:
		
		for i in range(N):
			line = f.readline().strip('\n')
			# print(line)
			y.append(line)
					
	return np.array(y)

def preprocess(el):
	translator=str.maketrans('','',string.punctuation)
	return "".join(el.translate(translator).lower().split(' '))

def get_label_voc(y):
	classes = {}
	ind = 1
	for i in range(len(y)):
		if y[i] not in classes:
			classes[y[i]] = ind
			ind += 1
	return classes 

def code_y(y, voc):
	return np.array([voc[w] for w in y])

def preprocess_features(X_train, X_val):
    oneh = OneHotEncoder()

    # encoded cities
    cities_train = oneh.fit_transform(X_train[:,0].reshape(-1,1))
    #cities_val = oneh.transform(X_val[:,0].reshape(-1,1))

    # encoded categ
    categ_train = oneh.fit_transform(X_train[:, 1].reshape(-1,1))
    #categ_val = oneh.transform(X_val[:,1].reshape(-1,1))

    # corpus of headlines; ony third input
    Xh = []

    for i in range(X_train.shape[0]):
        Xh.append(X_train[i][2])

    # print(Xh)

    xh_val = []
    for i in range(X_val.shape[0]):
        xh_val.append(X_val[i][2])

    # print(Xh)
    
    # fit tfidf to train only as well
    tfidf = TfidfVectorizer(
        max_df=0.95, 
        min_df=2, 
        max_features=100, 
        stop_words='english')

    xtr = tfidf.fit_transform(Xh)
    #print(xtr)
    # print(xtr.shape)
    xval = tfidf.transform(xh_val)

    
    features_train = hstack([xtr, categ_train, cities_train])

    return features_train



def main():
    X = load_data_inp('input00.txt')
    N = X.shape[0]
    y = load_data_out('output00.txt', N)

    classes = get_label_voc(y)
    # print(classes)
    classes_inv = dict([(v, k) for (k,v) in classes.items()])
    # print(classes_inv)
    # print(y)
    # print(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
    	test_size=0.2, random_state=42)
    

    train_labels= code_y(y_train, classes)
    test_labels = code_y(y_val, classes)

    features_train = preprocess_features(X_train, X_val)


    # Build feed forward classifier
    net = MLPClassifier(
        hidden_layer_sizes=(100, 100), 
        random_state=42,
        early_stopping=True,
        batch_size=32,
        alpha=0.001,
        validation_fraction=0.2
        )
    # print(xtr.shape)
    # print(train_labels.shape)
    net.fit(features_train, train_labels)
    preds = net.predict(features_train)

    # print(preds)
    # acc = accuracy_score(y_valid, preds, normalize=True)
    # print(acc)
    # pr = precision_recall_fscore_support(valid_labels, preds)
    # print(pr)

    for i in range(len(preds)):
    	# print(preds[i])
    	print(classes_inv[preds[i]])
    



if __name__ == '__main__':
    main()
