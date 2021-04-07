# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:25:25 2019

@author: Sophie
"""
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import matplotlib.pylab as plt
from nltk.stem import PorterStemmer

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



def sentence_count(str):
    # sentence = 0
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
    number_of_sentences = sent_tokenize(str)
    sentence=len(number_of_sentences)
    
    return sentence


def character_count(str):
    count=0
    for c in str:
        if c.isspace() != True :
            count += 1
    return count

# def process(filename,search_input):
#     xml = ET.parse(filename)  
#     detail = []
    
#     for article in xml.findall('.//Article'):
#         article_array = []
#         article_array.append(article.find('ArticleTitle').text)
    
#         abstract_texts = article.findall('.//AbstractText')
#         abstract_array = ''
#         for abstract_text in abstract_texts:
#             abstract_array += abstract_text.text
#         article_array.append(abstract_array)
#         detail.append(article_array)
def process(filename,search_input):
    xml = ET.parse(filename)
    detail = []
    
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
                    if 'Label' in abstract_text.attrib:
                        Label = abstract_text.attrib["Label"]
                        #print(Label)
                    
                        abstract_array +="\n"+Label+"\n"+''.join(abstract_text.itertext())
#                        print(abstract_array)
                #print(abstract_texts[0].text)
                
                article_array.append(abstract_array)
                #print(article_array)
        detail.append(article_array)
        
        #print(detail[0][1])
        
    #search_input=input("請輸入資料:")  
    #word_count=0  
    
    xml_per_detail = []
    for i in range(0,1):
        if search_input.lower() in detail[i][0].lower():
            #print('in!')
            x,y= zip(*word_count(detail[i][1]))
            a=dict({'id':str(i+1),'ARTITLE':str(detail[i][0]),'ABREST':str(detail[i][1]),'PER_CHARACTER':str(character_count(detail[i][1])),'WORDS':str(len(detail[i][1].split())),'PER_WORD': str(word_count(detail[i][1])),'PER_WORD_stem': str(word_stem(detail[i][1])),'SENTENCE': str(sentence_count(detail[i][1])),'zipf1_x':list(x),'zipf1_y':list(y)})
            #print(type(list(y)))
            #print("ID:" + str(i) + "\nARTITLE:" + str(detail[i][0]) + "\nABREST:" + str(detail[i][1]))
            #print("PER_CHARACTER:"+str(len(detail[i][1]))+"\nWORDS:"+str(len(detail[i][1].split()))+"PER_WORD:"+ str(word_count(detail[i][1]))) 
            
            
#            print("1.")
#            print(word_count(detail[i][1]))
#            print(word_stem(detail[i][1]))
#            
#            word_dic=word_count(detail[i][1])
            x,y= zip(*word_count(detail[i][1]))
            #plt.plot(x, y)
            #plt.show()
#            
#            stem_dic=word_stem(detail[i][1])
#            x, y = zip(*stem_dic)
#            plt.plot(x, y)
#            plt.show()
            
            
            xml_per_detail.append(a)
            
    return xml_per_detail
            
process('pubmed_hw2.xml','of')            
            
            #print("ID:" + str(i) + "\nARTITLE:" + str(detail[i][0]) + "\nABREST:" + str(detail[i][1]))
            #print("PER_CHARACTER:"+str(len(detail[i][1]))+"\nWORDS:"+str(len(detail[i][1].split()))+"PER_WORD:"+ str(word_count(detail[i][1])))  
    
    
def mark_span(o_string,keyword):
    keyword=keyword.lower()
    rep='<span style="color:orange;">'+keyword+'</span>'
    o_string=o_string.replace(keyword,rep)

    return o_string


#for neighbor in root.iter('neighbor'):
#     print neighbor.attrib

#def minimumEditDistance(first, second): 
#    matrix = np.zeros((len(first)+1,len(second)+1), dtype=np.int)
#    for i in range(len(first)+1): 
#        for j in range(len(second)+1): 
#            if i == 0:  
#                matrix[i][j] = j  
#            elif j == 0: 
#                matrix[i][j] = i
#            else: 
#                matrix[i][j] = min(matrix[i][j-1] + 1,  
#                                   matrix[i-1][j] + 1,        
#                                   matrix[i-1][j-1] + 2 if first[i-1] != second[j-1] else matrix[i-1][j-1] + 0)     
#                                   # Adjusted the cost accordinly, insertion = 1, deletion=1 and substitution=2
#    return matrix[len(first)][len(second)]  # Returning the final       
#print(minimumEditDistance('A','B'))        
