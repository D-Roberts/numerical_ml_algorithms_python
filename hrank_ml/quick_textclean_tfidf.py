import os 
import json
import string

import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    path = '.'
    files = os.listdir(path)
    
    X = []
    translator=str.maketrans('','',string.punctuation)
    N = int(input())
    for i in range(N):
        line = input()
        j = json.loads(line)['excerpt']
        # print(len(j.split()))
        w = [w1.translate(translator).lower() for w1 in j.split() if w1.isalpha()]
        # leave remove stopwords for tfidf to do
        # print(' '.join(w))
        X.append(' '.join(w))

    # print(X)
        
    tf = TfidfVectorizer()
    X1 = tf.fit_transform(np.array(X))
    X2 = X1.toarray()
    print(X2)

    

if __name__ == '__main__':
    main()