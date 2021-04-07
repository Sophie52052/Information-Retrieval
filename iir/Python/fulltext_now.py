# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:03:49 2019

@author: Sophie
"""

import xml.etree.ElementTree as ET
tree = ET.parse('pubmed19n0001.xml')
tree = ET.parse('A1.xml')
root = tree.getroot()

for element in tree.findall( 'PubmedArticle/MedlineCitation/Article/ArticleTitle' ):
    if 'assay' in element.text.lower():
        print(element.text)
        

for element in tree.findall( 'PubmedArticle/MedlineCitation/Article' ):
    X=element.find("ArticleTitle").text.lower()
    print(X)
    
for element in tree.findall( 'PubmedArticle/MedlineCitation/Article/Abstract' ):
    X=element.find("AbstractText").text.lower()
    print(X)
        
    
    
    
for element in tree.findall( 'PubmedArticle/MedlineCitation/Article' ):
    title=element.find("ArticleTitle").text.lower()
    if 'assay' in title:
        print(title)
        
        
        
for element in tree.findall( 'PubmedArticle/MedlineCitation/Article' ):
    context=element.find("Abstract/AbstractText").text.lower()
    print(context)        
        
        
for element in tree.findall( 'PubmedArticle/MedlineCitation/Article' ):
    X=element.find("Abstract/AbstractText").text.lower()
    print(X)
    
    
    
    context=element.find("Abstract/AbstractText").text.lower()
    
    
    
import xml.etree.ElementTree as ET
tree = ET.parse('pubmed19n0001.xml')
tree = ET.parse('A1.xml')
tree = ET.parse('pubmed_result.xml')
tree.findall('PubmedArticle')
tree.getroot()
root = tree.getroot()

   
   for child_of_root in root:
    for A in child_of_root:
        print (A.tag, A.attrib)
        
        
root[0][3][1].text
Out[76]: '[beta-blockers in hypertension with renal failure].'

name = input('請輸入檔名：')

#tree.find("Books/Book[ArticleTitle='Formate assay in body fluids: application in methanol poisoning.']") 
## always use '' in the right side of the comparison
#
#tree.find("Books/Book[@id='5']")
## searches with xml attributes must have '@' before the name

pubmed_result


#ALL-XML

#PubmedArticle/MedlineCitation/Article/ArticleTitle
for element in tree.findall( './/ArticleTitle' ):
    print(element.text)
    
    
#PubmedArticle/MedlineCitation/Article/Abstract/AbstractText
for context in tree.findall( './/AbstractText' ):
    print(context.text)

for element in tree.findall( 'PubmedArticle/MedlineCitation/Article' ):
    print(element.find("ArticleTitle").text)
    
for element in tree.findall( './/PMID' ):
    print(element.text)

#.//ArticleTitle

A=tree.find( "PubmedArticle/MedlineCitation/Article[ArticleTitle='Formate assay in body fluids: application in methanol poisoning.']" )

A


if 'assay' in A[0].text:
    print(A[0].text)
    
for element in tree.findall( './/ArticleTitle' ):
    if 'cancer' in element.text:
        print(element.text)
    else :print("no")
    
    
for element in tree.findall( './/ArticleTitle' ):
    A=element.text
    
    if 'assay' in A:
        print(A)

## A1
#for artical in tree.findall( 'MedlineCitation/Article/ArticleTitle' ):
#    print(artical.text)
#
#for context in tree.findall( 'MedlineCitation/Article/Abstract/AbstractText' ):
#    print(context.text)
        
        
for element in tree.findall( './/ArticleTitle' ):
    A=element.text
    for context in tree.findall( './/PMID' ):
        B=context.text
    if 'assay' in A:
        print(B)
        print(A)
        
        
        
        
for context in tree.findall( './/ArticleId[@IdType="pubmed"]' ):
    B=context.text
    print(B)


#-----------------------------------------------------------------

     
        
jfile = open('j1.json','r',encoding="utf-8")
jstr = jfile.read()
jdata = json.loads(str(jstr))

data[0]["created_at"]


import json

data = [json.loads(line) for line in open('30.json', 'r')]
delete myObj.test.key1;







#----------------------------不分
#算字(個別)
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

print( word_count(detail[0]["text"]))   


#算字(總數)
word_count=0
word_count+=len(detail[0]["text"].split())
print(word_count)