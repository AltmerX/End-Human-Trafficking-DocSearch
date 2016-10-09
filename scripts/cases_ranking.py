# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 21:12:25 2016

@author: oser
"""

import numpy as np
import pandas as pd
import random

'''
dict_test = {'1':4, '2':3, '3': 6, '4':2, '5':5, '6':1}
print(sorted(dict_test, key=dict_test.__getitem__))

'''

def cases_rank(tf_idf, w_emb, pattern, method = 1):
    tf_idf = tf_idf.sub(tf_idf.mean(axis=1), axis=0)
    median = tf_idf.median(axis=1)
    #print(median)
    corr_ceil = 0.8
    corr_matrix = w_emb.corr().as_matrix()
    #print(corr_matrix[1,0])
    words = list(w_emb.columns)
    words_dict = {}
    for i in range(len(words)):
        words_dict[words[i]] = i
    ext_pattern = pattern
    dict_ES = {}
    for w1 in pattern:
        if (w1 in words):
            for w2 in words:
                if ((corr_matrix[words_dict[w1], words_dict[w2]] >= corr_ceil) &( w2 not in pattern)):
                    ext_pattern.append(w2)
    if (method == 1)   :            
        for case_id in list(tf_idf.index):
            dict_word_value = tf_idf.loc[case_id].to_dict()
            ranked_keys_list = sorted(dict_word_value, key=dict_word_value.__getitem__, reverse = True)
            value_sum = 0
            ES = 0
            for w in ranked_keys_list:
                if w in ext_pattern:
                    value_sum = value_sum + dict_word_value[w]
                else:
                    value_sum = value_sum - dict_word_value[w]   
                if(value_sum > ES):
                    ES = value_sum
                    
            dict_ES[case_id] = ES
             
        return [(x, dict_ES[x]) for x in sorted(dict_ES, key=dict_ES.__getitem__, reverse = True)]
    else:   
        i = 0
        ext_pattern_in = [x for x in ext_pattern if x in list(tf_idf.columns)]
        print[median[i], [tf_idf[x][i] for x in list(tf_idf.columns)]]
        over_median_pattern = len([x for x in ext_pattern_in if tf_idf[x][i] >= median[i]])
        under_median_pattern = len([x for x in ext_pattern_in if tf_idf[x][i] < median[i]])
        over_median_rest = len([x for x in list(tf_idf.columns) if (tf_idf[x][i] >= median[i]) & (x not in ext_pattern_in)])
        under_median_rest = len([x for x in list(tf_idf.columns) if (tf_idf[x][i] < median[i]) & (x not in ext_pattern_in)])
        print(over_median_pattern, under_median_pattern, over_median_rest, under_median_rest)
        
        
        '''
        dict_word_value = tf_idf.iloc[i].to_dict()
        ranked_keys_list = sorted(dict_word_value, key=dict_word_value.__getitem__, reverse = True)
        value_sum = 0
        ES = 0
        for w in ranked_keys_list:
            if w in ext_pattern:
                value_sum = value_sum + dict_word_value[w]
            else:
                value_sum = value_sum - dict_word_value[w]   
            if(value_sum > ES):
                ES = value_sum
                
        dict_ES[i] = ES
         
    return [(x, dict_ES[x]) for x in sorted(dict_ES, key=dict_ES.__getitem__, reverse = True)]
    '''
"""
tf_idf = pd.read_csv('tf_idf.csv')
w_emb = pd.read_csv('word_embbeding_escort_data.csv')

list_for_pattern =  random.sample(range(w_emb.shape[1]), 7)  

pattern = list(w_emb.columns[    random.sample(range(w_emb.shape[1]), 7)  ])

print(pattern_find(tf_idf, w_emb, pattern))
"""