# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:08:43 2016

@author: julmaud
"""

import os
import sys
import operator
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from elasticsearch import Elasticsearch    
import pandas as pd
import numpy as np
import word_vectorization
import cases_ranking
import nltk
import string
from nltk.stem.porter import PorterStemmer

#tokenize functions

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
   stemmed = []
   for item in tokens:
       stemmed.append(stemmer.stem(item))
   return stemmed
   
def tokenize(text):
   text = "".join([ch for ch in text if ch not in string.punctuation])
   tokens = nltk.word_tokenize(text.lower())
   stems = stem_tokens(tokens, stemmer)
   return stems

#Data Collection and Extraction
def format_data():
    case_ids = ['African-american', 'Asian', 'European', 'South American', 'US']
    #case_ids = [str(k) for k in range(1,6)]
    #case_ids = ['Season-'+str(k) for k in range(1,20)]
    docs_per_case = {}
    for case_id in case_ids:
        docs_per_case[case_id] = []
        case_path = '../clean_escort_ads/'+case_id+'/'
        
        for filename in os.listdir(case_path):
            #if filename[0] is '.':
            #    continue
            file_path = case_path+filename
            f = open(file_path, 'r')
            text = f.read()
            docs_per_case[case_id].append({'title': filename, 'text': text.lower()})
        
        cases_as_docs = {}
        for (case_id, list_of_docs) in docs_per_case.iteritems():
            case_text = ''
            for doc in list_of_docs:
                case_text += ' '+doc['text']
            cases_as_docs[case_id] = case_text
    return cases_as_docs, docs_per_case
    
def format_data_bible():
    case_ids =['Bible']
    docs_per_case = {}
    for case_id in case_ids:
        docs_per_case[case_id] = []
        case_path = '../'+case_id+'/'
        
        for filename in os.listdir(case_path):
            #if filename[0] is '.':
            #    continue
            file_path = case_path+filename
            f = open(file_path, 'r')
            text = f.read()
            docs_per_case[case_id].append({'title': filename, 'text': text.lower()})
        
        cases_as_docs = {}
        for (case_id, list_of_docs) in docs_per_case.iteritems():
            case_text = ''
            for doc in list_of_docs:
                case_text += ' '+doc['text']
            cases_as_docs[case_id] = case_text
    return cases_as_docs, docs_per_case

#Prior analysis

def create_tfidf_vector(cases_as_doc):
    #input: {case_id: text}
    indexed_casecontent = {cpt:{'case_id':case_id,'content':text} for (cpt,(case_id, text)) in zip(range(len(cases_as_doc)),cases_as_doc.iteritems())}
    tf = TfidfVectorizer(analyzer='word', tokenizer = tokenize, ngram_range=(1,1), min_df = 0, stop_words = 'english')
    bloblist=[dic['content']for (cpt,dic) in indexed_casecontent.iteritems()]
    tfidf_matrix =  tf.fit_transform(bloblist)
    feature_names = tf.get_feature_names()
    dic_case_vector = {indexed_casecontent[cpt]['case_id']:{'phrases':feature_names, 'scores':tfidf_matrix[cpt,].toarray().tolist()[0]} for cpt in range(len(cases_as_doc))}
    return dic_case_vector
    
#Elasticsearch setuo
    
def setup_es():
    es_host = {"host" : "localhost", "port" : 9200}
    es = Elasticsearch(hosts = [es_host])
    return es
    
    
def create_index(es, index_name, request_body):
    if es.indices.exists(index = index_name):
        es.indices.delete(index = index_name)
    print("creating '%s' index..." % (index_name))
    res = es.indices.create(index = index_name, body = request_body)
    print(" response: '%s'" % (res))
    

def add_case_to_cases_index(es, case_id, phrases, scores, text):
    case = {"case_id":case_id,
            "phrases": phrases,
            "scores": scores,
            "text": text
            }
    es.index(index = 'cases', doc_type = 'case', body=case)


def add_document_to_documents_index(es, doc_title, case_id, text):
    document = {"doc_title":doc_title,
                "case_id":case_id,
                "text": text
                }
    es.index(index = 'documents', doc_type = 'document', body=document)


def build_elasticsearch_database(es, cases_as_docs, docs_per_case):
    #cases_as_docs: {case_id: text}
    #docs_per_case: {case_id: [{'title': title, 'text': text}]}
    
    #create_index(es, index_name_documents, request_body_documents)
    #create_index(es, index_name_cases, request_body_cases)

    dic_case_vector = create_tfidf_vector(cases_as_docs)
    for case_id in cases_as_docs.keys():
        add_case_to_cases_index(es, case_id, dic_case_vector[case_id]['phrases'], dic_case_vector[case_id]['scores'], cases_as_docs[case_id])
            
    for (case_id, docs) in docs_per_case.iteritems():
        for doc in docs:
            add_document_to_documents_index(es, doc['title'], case_id, doc['text'])
    es.indices.refresh('_all')


def get_ranked_patterns(es, case_id, number):
    res = es.search(index="cases", doc_type="case", body={"query": {"match": {"case_id": case_id}}})
    phrases = res['hits']['hits'][0]['_source']['phrases']
    scores = res['hits']['hits'][0]['_source']['scores']
    word_freq = {phrases[cpt]:scores[cpt] for cpt in range(len(phrases))}
    word_freq_sorted = sorted(word_freq.items(), key = operator.itemgetter(1), reverse = True)
    ranked_pattern = [word_freq_sorted[cpt] for cpt in range(number)]
    return ranked_pattern
    
    
def get_ranked_cases(es, pattern, number):
    res = es.search(index="cases", doc_type="case", body={"query": {"match_all": {}}})
    case_ids = set([res['hits']['hits'][k]['_source']['case_id'] for k in range(len(res['hits']['hits']))])
    phrases = res['hits']['hits'][0]['_source']['phrases']
    text_concat = ''
    tf_idf = pd.DataFrame(np.random.randn(len(case_ids), len(phrases)), columns=phrases, index=case_ids)
    for case_id in case_ids:
        res_vect = es.search(index="cases", doc_type="case", body={"query": {"match": {"case_id": case_id}}})
        tf_idf.loc[case_id] = res_vect['hits']['hits'][0]['_source']['scores']
        text_concat += ' '+res_vect['hits']['hits'][0]['_source']['text']
        
    w_emb = word_vectorization.word_embedding(text_concat)
    rk = cases_ranking.cases_rank(tf_idf, w_emb, pattern, method = 1)[:number]
    
    return rk
    
def get_ranked_files(es, pattern, case_id, number):
    
    query ={
    "query": {
        "match": {
            "text": pattern
        }
    }
    }
    res = es.search(index="documents", doc_type="document", body=query)
    nb_match = len(res['hits']['hits'])
    rk = []
    for k in range(nb_match):
        if  res['hits']['hits'][k]['_source']['case_id'].encode('utf-8') != case_id:
            continue
        rk.append(res['hits']['hits'][k]['_source']['doc_title'])
    return rk[:min(number,len(rk))]
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

