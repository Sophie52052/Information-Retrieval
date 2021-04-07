# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:25:25 2019

@author: Sophie
"""
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize,TweetTokenizer
from collections import Counter
from nltk.stem import PorterStemmer
import numpy as np
import editdistance
from autocorrect import spell
import matplotlib.pyplot as plt, mpld3
from mpld3 import save_json, fig_to_html, plugins
from nltk.corpus import stopwords
import math 
from sklearn.feature_extraction.text import TfidfVectorizer

def word_count(str):
    counts = dict()
    words = word_tokenize(str.lower())
    words1 = nltk.Text(words)
    words = [w.lower() for w in words1 if w.isalpha()]
    #print(words)
    #words = str.split()
    
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in words if not w in stop_words]
    
    for w in words: 
        if w not in stop_words: 
            filtered_sentence.append(w)
            #print(filtered_sentence)      
    

    for word in filtered_sentence:
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




def process(filename,search_input,search_input1):
     #search_input1="cancer"
     xml = ET.parse('mesh_per.xml')
    
###########################頭
     title_all=[]
     for title in xml.findall('DescriptorRecord/DescriptorName'):
       article_array = []
       article_array.append(title.find('String').text)
       title_all.append(article_array)
     print(title_all)
        
        
    #xml = ET.parse('mesh_per.xml')
     mesh_dict = []
     for term in xml.findall('.//ConceptList'):
        mesh_term=term.findall('.//String')
        mesh_term_all = []
        for per in mesh_term:
            per_mesh=''
            per_mesh+=per_mesh+per.text
            mesh_term_all.append(per_mesh)
        mesh_dict.append(mesh_term_all)

     for i in range(0,len(title_all)):
        if search_input in title_all[i]:
            mesh_term=mesh_dict[i]
            #print(mesh_dict[i])
            #print(i)
     print(mesh_term)
############################


     xml = ET.parse(filename)  
     detail = []
     abstract_array_all=[]    
     for article in xml.findall('.//Article'):
         article_array = []
         article_array.append(article.find('ArticleTitle').text)
    
         abstract_texts = article.findall('.//AbstractText')
         abstract_array = ''
         for abstract_text in abstract_texts:
             abstract_array += abstract_text.text
         article_array.append(abstract_array)
         abstract_array_all.append(abstract_array)
         detail.append(article_array)

##############
     mesh_array = []
     
     for MeshHeading in xml.findall('.//MeshHeadingList'):
         mesh_location=MeshHeading.findall('.//DescriptorName')
         mesh_per_total=[]
         for per_mesh in mesh_location :
             mesh_per=''
             
             mesh_per=mesh_per+ per_mesh.text
             mesh_per_total.append(mesh_per)
         mesh_array.append(mesh_per_total)
##########

     corpus=abstract_array_all
     vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
     tfidf = vect.fit_transform(corpus)                                                                                                                                                                                                                       
     pairwise_similarity = tfidf * tfidf.T
        
     A=pairwise_similarity.toarray() 
     #A=A[A==1]=0
     similarity_array=[]
     index_array=[]
     for m in range(0,len(A)):
         f=[]
         b=list(A[m])
         b=[0 if x>=0.99 else x for x in b]
         similarity_array.append(b)
         c=max(similarity_array[m])
         #print(c)
         index = similarity_array[m].index(c)
         index_true=index+1
         f.append(index_true)
         f.append(c)
         index_array.append(f)         
    


# def process(filename,search_input,search_input2):
#     xml = ET.parse(filename)
#     detail = []
    
#     for article in xml.findall('.//Article'):
#         article_array = []
#         article_array.append(article.find('ArticleTitle').text)
    
#         abstract_texts = article.findall('.//AbstractText')       
#         abstract_array = '' 
        
#         #print(abstract_texts)
#         #print(len(abstract_texts))
#         #print(len(abstract_texts))
#         if abstract_texts is None:
#             abstract_array.append('')
            
#         else:
#                 for abstract_text in abstract_texts:
#                     if 'Label' in abstract_text.attrib:
#                         Label = abstract_text.attrib["Label"]
#                         #print(Label)
                    
#                         abstract_array +=Label+"<br><br>"+''.join(abstract_text.itertext())+"<br><br>"
# #                        print(abstract_array)
#                 #print(abstract_texts[0].text)
                
#                 article_array.append(abstract_array)
# #                        print(article_array)
#         detail.append(article_array)
        
#         #print(detail[0][1])
        
#     #search_input=input("請輸入資料:")  
#     #word_count=0  
     xml_per_detail = []
     per_num=0
     tfidf_array = []
     tfidf1_array=[]
     per_num_s=0
     tfidf_t_array=[]
     tfidf_t_array1=[] 
     tfidf2_array=[]
     tfidf_t_array2=[]

     xml_per_detail = []
     #for i in range(0,len(detail)):
     for i in range(0,len(detail)):
      for j in range(0,len(mesh_array[i])):
         #len(detail)
        sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        #if search_input.lower() in detail[i][0].lower():
        if search_input.lower() in mesh_array[i][j].lower():
            total=0
            num=0
            per_freq=word_count(detail[i][1])
            
            A=0
            total_s=0
            per_count=0
            per_freq_s=sen_tokenizer.tokenize(detail[i][1])
            A=len(per_freq_s)
            tfidf_p_array=[]
            tfidf_p_array1=[]
            tfidf_p_array2=[]

            
            for j in range(0,len(per_freq)):
                #if search_input == per_freq[j][0]:
                if search_input1.lower() == per_freq[j][0].lower():
                    per_num+=1
                    num=per_freq[j][1]
                total+=per_freq[j][1]
                df=num / total
                
                df1=1
                
                
                if total==0 or num==0:
                    idf1=0
                    df2_temp=0
                else:
                    a=num / total
                    idf1=math.log(a)
                    
                    df2_temp=math.log(a)
#                if num==0:
                df2=1+df2_temp  
#                else:                    
#                    df1=1
                if per_num==0:
                    idf_temp=0
                    idf_temp2=0
                    idf=0#
                    idf2=0#
                else:
                    idf_temp=len(detail)/per_num
                    idf_temp2=len(detail)/per_num
                    idf=math.log(idf_temp)#
                    idf2=math.log(idf_temp2)#
            
            #idf=math.log(idf_temp)
            #idf2=math.log(idf_temp2)
            tfidf=round(df*idf, 3)
            tfidf1=round(df1+idf1, 3)
            tfidf2=round(df2*idf2, 3)
            tfidf_array.append(tfidf)
            tfidf1_array.append(tfidf1)
            tfidf2_array.append(tfidf2)
            
            for j_s in range(0,A):
                per_count_temp=word_count(per_freq_s[j_s])
                
                for k in range(0,len(per_count_temp)):
                    #if search_input == per_count_temp[k][0]:
                    if search_input1 == per_count_temp[k][0].lower():
                         per_num_s+=1
                         per_count=(per_count_temp[k][1])/2 #num
                                         
                    total_s+=per_count
                         #print(total_s)
                    df_s1=1
                    if total_s ==0 :
                        df_s=0
                        idf_s1=0
                        df_s_temp2=0
                        
                    else :
                        df_s=per_count/total_s
                        df_s_temp2=math.log(per_count/total_s)
                        idf_s_temp1=per_count/total_s
                        idf_s1=math.log(idf_s_temp1)
                        
                    if per_num_s==0:
                        idf_s_temp=0
                        idf_s_temp2=0
                        idf_s=0#
                        idf_s2=0#
                    else:
                        idf_s_temp=len(detail)/per_num_s
                        idf_s_temp2=len(detail)/per_num_s
                        idf_s=math.log(idf_s_temp)#
                        idf_s2=math.log(idf_s_temp2)#
                #idf_s=math.log(idf_s_temp)
                #idf_s2=math.log(idf_s_temp2)
                df_s2=1+df_s_temp2
                tfidf_s=round(df_s*idf_s, 3)
                tfidf_s1=round(df_s1+idf_s1, 3)
                tfidf_s2=round(df_s2*idf_s2, 3)
                tfidf_p_array.append(tfidf_s)
                tfidf_p_array1.append(tfidf_s1)
                tfidf_p_array2.append(tfidf_s2)
            #print(tfidf_p_array)
            tfidf_t_array.append(tfidf_p_array)
            tfidf_t_array1.append(tfidf_p_array1)
            tfidf_t_array2.append(tfidf_p_array2)

            #print('in!')
            x,y= zip(*word_count(detail[i][1]))
            x2,y2= zip(*word_stem(detail[i][1]))
            zipf_id=str(i)
            zipf2_id=str(i+1000)

            a=dict({'id':zipf_id,'id2':zipf2_id,
                    'ARTITLE':mark_span(str(detail[i][0]),search_input1),
                    'ABREST':mark_span(str(detail[i][1]),search_input1),
                    'PER_CHARACTER':str(character_count(detail[i][1])),
                    'WORDS':str(len(detail[i][1].split())),
                    'PER_WORD': str(word_count(detail[i][1])), 
                    'PER_WORD_stem': str(word_stem(detail[i][1])),
                    'SENTENCE': str(sentence_count(detail[i][1])),
                    'zipf1_x':list(x),
                    'zipf1_y':list(y),
                    'zipf2_x':list(x2),
                    'zipf2_y':list(y2),
                    'tfidf':tfidf_array[i],
                    'tfidf_p':tfidf_t_array[i],
                    'tfidf1':tfidf1_array[i],
                    'tfidf_p1':tfidf_t_array1[i],
                    'tfidf2':tfidf2_array[i],
                    'tfidf_p2':tfidf_t_array2[i],
                    'similarity_index':index_array[i][0],
                    'similarity_value':index_array[i][1],
                    'mesh_word':mesh_array[i],
                    'mesh_term':mesh_term,
                    })
            #a=dict({'id':zipf_id,'id2':zipf2_id,'ARTITLE':mark_span(str(detail[i][0]),search_input,search_input2),'ABREST':mark_span(str(detail[i][1]),search_input,search_input2),'PER_CHARACTER':str(character_count(detail[i][1])),'WORDS':str(len(detail[i][1].split())),'PER_WORD': str(word_count(detail[i][1])), 'PER_WORD_stem': str(word_stem(detail[i][1])),'SENTENCE': str(sentence_count(detail[i][1])),'zipf1_x':list(x),'zipf1_y':list(y),'zipf2_x':list(x2),'zipf2_y':list(y2),'tfidf':tfidf_array[i],'tfidf_p':tfidf_t_array[i],'tfidf1':tfidf1_array[i],'tfidf_p1':tfidf_t_array1[i],'tfidf2':tfidf2_array[i],'tfidf_p2':tfidf_t_array2[i],'similarity_index':index_array[i][0],'similarity_value':index_array[i][1],})
            
            #print(a)
            #print("ID:" + str(i) + "\nARTITLE:" + str(detail[i][0]) + "\nABREST:" + str(detail[i][1]))
            #print("PER_CHARACTER:"+str(len(detail[i][1]))+"\nWORDS:"+str(len(detail[i][1].split()))+"PER_WORD:"+ str(word_count(detail[i][1]))) 
            #            word_dic=word_count(detail[i][1])

            
            # x, y = zip(*word_count(detail[i][1]))
            # plt.plot(x, y)
            # plt.show()
           
            # stem_dic=word_stem(detail[i][1])
            # x, y = zip(*stem_dic)
            # plt.plot(x, y)
            # plt.show()
            
            xml_per_detail.append(a)
            x=xml_per_detail
            y=sorted(range(len(x)), key = lambda k:x[k]['tfidf'], reverse=True)
            x = [x[i] for i in y]
            xml_per_detail=x
            
     return xml_per_detail
            
            
            #print("ID:" + str(i) + "\nARTITLE:" + str(detail[i][0]) + "\nABREST:" + str(detail[i][1]))
            #print("PER_CHARACTER:"+str(len(detail[i][1]))+"\nWORDS:"+str(len(detail[i][1].split()))+"PER_WORD:"+ str(word_count(detail[i][1])))  
    
    
#def mark_span(o_string,keyword,keyword2):
def mark_span(o_string,keyword):
    keyword=keyword.lower()
    rep='<span style="color:orange;">'+keyword+'</span>'
    #rep1='<span style="color:pink;">'+keyword2+'</span>'
    o_string=o_string.replace(keyword,rep)
    #o_string2=o_string.replace(keyword2,rep1)
    return o_string

# def minimumEditDistance(keyword, keyword2): 
#     edit_distance=editdistance.eval(keyword, keyword2)                           
#     return edit_distance

# def autocorrect(keyword, keyword2): 
#     spell1=spell(keyword)
#     spell2=spell(keyword2)             
#     return spell1,spell2
  

# def minimumEditDistance(keyword, keyword2): 
#     matrix = np.zeros((len(keyword)+1,len(keyword2)+1), dtype=np.int)
    
#     for i in range(len(keyword)+1): 
#         for j in range(len(keyword2)+1): 
#             if i == 0:  
#                 matrix[i][j] = j  
#             elif j == 0: 
#                 matrix[i][j] = i
#             else: 
#                 matrix[i][j] = min(matrix[i][j-1] + 1,  
#                                    matrix[i-1][j] + 1,        
#                                    matrix[i-1][j-1] + 2 if keyword[i-1] != keyword2[j-1] else matrix[i-1][j-1] + 0)     
#     sys.setrecursionlimit(1000000)
#     print(minimumEditDistance(keyword,keyword2))                            
#     return matrix[len(keyword)][len(keyword2)] 
 
        
        
        
