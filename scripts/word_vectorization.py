# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gensim
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

stop = stopwords.words('english')

def word_embedding(txt):
    sent_tokenize_list = sent_tokenize(txt)
    sent_tokenize_list = [ "".join([ch for ch in x if ch not in string.punctuation]) for x in sent_tokenize_list]
    word_tokens = [word_tokenize(x.lower()) for x in sent_tokenize_list]
    word_tokens = [[x for x in y if x not in stop] for y in word_tokens]
    model = gensim.models.Word2Vec(word_tokens, size=300, window=5, min_count=1, workers=4)
    vocab = list(model.vocab.keys())
    list_columns = [model[x] for x in vocab]
    df = pd.DataFrame(list_columns).T
    df.columns = vocab
    
    return df    
    
    
    
    
    
    
    
    """
    print(word_tokenize(sent_tokenize_list[0]))
    print((model[vocab[0]]))
    print(model[vocab[0]])
    df.to_csv("word_embbeding_escort_data.csv", encoding='utf-8')
    #print(df.head())
    
    #plt.matshow(df.transpose().corr())
    
    pca = PCA(n_components=2)
           
    
    df_pca = pca.fit_transform(df)
    plt.scatter(df_pca[:,0], df_pca[:,1])
    plt.show()

    """