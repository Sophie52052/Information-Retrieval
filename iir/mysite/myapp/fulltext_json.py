# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:48:34 2019

@author: Sophie
"""

import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import matplotlib.pylab as plt
from nltk.stem import PorterStemmer
import numpy as np
import editdistance
from autocorrect import spell

def sentence_count(str):
    # sentence = 1
    # seen_end = False
    # sentence_end = {'?', '!', '.'}
    # yourstring=str
    # for c in yourstring:
    #     if c in sentence_end:
    #          if not seen_end:
    #              seen_end = True
    #              sentence += 1
    #          continue
    #     seen_end = False
    # if sentence=="0" :
    #     sentence=1
    sentences=str
    
    number_of_sentences = sent_tokenize(sentences)
    sentence=len(number_of_sentences)
    return sentence

def word_count(str):
    counts = dict()
    words = word_tokenize(str)
    #words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    counts = Counter(counts).most_common()
    #return sorted(counts.items(), key=lambda d: d[1], reverse=True)
    return counts

def word_stem(str):
    counts = dict()
    words = word_tokenize(str)
    ps = PorterStemmer()

    for word in words:
        word=ps.stem(word)
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    counts = Counter(counts).most_common()
    #return sorted(counts.items(), key=lambda d: d[1], reverse=True)
    return counts

def character_count(str):
    count=0
    for c in str:
        if c.isspace() != True :
            count += 1
    return count

def process(filename,search_input,search_input2):
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
            x,y= zip(*word_count(detail[i]["text"]))
            x2,y2= zip(*word_stem(detail[i]["text"]))
            a=dict({'INDEX':str(i),'INDEX2':str(i+1000),'TEXT':mark_span(str(detail[i]["text"]),search_input,search_input2),'PER_CHARACTER':str(character_count(detail[i]["text"])),'WORDS':str(len(detail[i]["text"].split())),'PER_WORD': str(word_count(detail[i]["text"])), 'PER_WORD_stem': str(word_stem(detail[i]["text"])),'SENTENCE': str(sentence_count(detail[i]["text"])),'SENTENCE': str(sentence_count(detail[i]["text"])),'zipf1_x':list(x),'zipf1_y':list(y),'zipf2_x':list(x2),'zipf2_y':list(y2)})
            print(a)
            json_per_detail.append(a)
            #print("INDEX:" + str(i) + "\nID:" + str(detail[i]["id"]) + "\nTEXT:" + str(detail[i]["text"]) )    
            #print("PER_CHARACTER:"+str(len(detail[i]["text"]))+"\nWORDS:"+str(len(detail[i]["text"].split()))+"PER_WORD:"+ str(word_count(detail[i]["text"]))+"\n")
    print(len(json_per_detail))
    return json_per_detail      

#process('J2.json','how')   

def mark_span(o_string,keyword,keyword2):
    keyword=keyword.lower()
    rep='<span style="color:orange;">'+keyword+'</span>'
    rep1='<span style="color:pink;">'+keyword2+'</span>'
    o_string=o_string.replace(keyword,rep)
    o_string2=o_string.replace(keyword2,rep1)
    return o_string2

def minimumEditDistance(keyword, keyword2): 
    edit_distance=editdistance.eval(keyword, keyword2)                           
    return edit_distance

def autocorrect(keyword, keyword2): 
    spell1=spell(keyword)
    spell2=spell(keyword2)             
    return spell1,spell2


        




