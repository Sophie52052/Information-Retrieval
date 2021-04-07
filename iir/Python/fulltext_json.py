# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:48:34 2019

@author: Sophie
"""

import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_count(str):
#    sentence = 1
#    seen_end = False
#    sentence_end = {'?', '!', '.'}
#    yourstring=str
#    for c in yourstring:
#        if c in sentence_end:
#             if not seen_end:
#                 seen_end = True
#                 sentence += 1
#             continue
#        seen_end = False
#    if sentence=="0" :
#        sentence=1
    sentences=str
    sentence = sent_tokenize(sentences)
    
    
    return sentence

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

def process(filename,search_input):
    detail = json.loads(str(open(filename,'r',encoding="utf-8").read()))
#---------------------
#有,
#detail = [json.loads(line) for line in open('30.json', 'r')]

#data = []
#for line in open('30.json', 'r',encoding="utf-8"):
    #data.append(json.loads(line))

    data_array = []
    for msg in detail:
        per_data = []
        per_data.append(msg['ID'])
        per_data.append(msg['text'])
        data_array.append(data_array)


    json_per_detail = []
    for i in range(0,len(detail)):
        if search_input.lower() in detail[i]["text"].lower():
            a=dict({'INDEX':str(i),'TEXT':str(detail[i]["text"]),'PER_CHARACTER':str(len(detail[i]["text"])),'WORDS':str(len(detail[i]["text"].split())),'PER_WORD': str(word_count(detail[i]["text"])),'SENTENCE': str(sentence_count(detail[i]["text"]))})
            print(a)
            json_per_detail.append(a)
            #print("INDEX:" + str(i) + "\nID:" + str(detail[i]["id"]) + "\nTEXT:" + str(detail[i]["text"]) )    
            #print("PER_CHARACTER:"+str(len(detail[i]["text"]))+"\nWORDS:"+str(len(detail[i]["text"].split()))+"PER_WORD:"+ str(word_count(detail[i]["text"]))+"\n")
    print(len(json_per_detail))
    return json_per_detail        

process('tweet_dengue_ncku.json','dengue')   




