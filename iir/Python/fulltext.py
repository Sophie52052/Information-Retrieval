# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:25:25 2019

@author: Sophie
"""
import xml.etree.ElementTree as ET

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def sentence_count(str):
    sentence = 0
    seen_end = False
    sentence_end = {'?', '!', '.'}
    yourstring=str
    for c in yourstring:
        if c in sentence_end:
             if not seen_end:
                 seen_end = True
                 sentence += 1
             continue
        seen_end = False
    return sentence

def process(filename,search_input):
    xml = ET.parse(filename)  
    detail = []
    i=0
    X = search_input.split()
    print(len(X))
    for article in xml.findall('.//Article'):
        article_array = []
        article_array.append(article.find('ArticleTitle').text)
    
        abstract_texts = article.findall('.//AbstractText')       
        abstract_array = '' 
        
        #print(abstract_texts)
        #print(len(abstract_texts))
        #print(len(abstract_texts))
        if abstract_texts is None:
            abstract_array.append('')
            
        else:
                for abstract_text in abstract_texts:
                    
                    #print("1")
                    
                    abstract_array += ''.join(abstract_text.itertext())
                    #print(abstract_array)
                #print(abstract_texts[0].text)
        article_array.append(abstract_array)
        detail.append(article_array)
        #print(detail[0][1])
        
    #search_input=input("請輸入資料:")  
    #word_count=0  
    #search_input = "dengue"
    
    per_detail = []
    for i in range(0,10):
#        der = detail[i][1].lower()
#        der = detail[i][0].lower()
        for j in range(0,len(X)):
            if search_input.lower() in detail[i][1].lower():
                #print('in!')
                print(i)
                a=dict({'ID':str(i),'ARTITLE':str(detail[i][0]),'ABREST':str(detail[i][1]),'PER_CHARACTER':str(len(detail[i][1])),'WORDS':str(len(detail[i][1].split())),'PER_WORD': str(word_count(detail[i][1])),'SENTENCE': str(sentence_count(detail[i][1]))})
    #            print(a)
                per_detail.append(a)
            
    return per_detail
            
            
            
            #print("ID:" + str(i) + "\nARTITLE:" + str(detail[i][0]) + "\nABREST:" + str(detail[i][1]))
            #print("PER_CHARACTER:"+str(len(detail[i][1]))+"\nWORDS:"+str(len(detail[i][1].split()))+"PER_WORD:"+ str(word_count(detail[i][1])))  
    
    
process('pubmed_dengue3.xml','dengue fever')     
        
        
        
