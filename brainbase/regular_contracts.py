# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:44:01 2020

@author: Yury
"""

import re

#emaple of a contract
#https://anglofon.com/sample-consignment-contract?w
# https://www.kaggle.com/jpandeinge/nlp-analysis-of-pdf-documents
# http://theautomatic.net/2020/01/21/how-to-read-pdf-files-with-python/
txt='''This agreement is made and entered into by and between 'Abc & Co.' and 'Bcd LLC' for term 1 year starting from April 1, 2020,
hereinafter collectively referred to as the Parties. Samuel L. Jackson in the place (New York) and on the date written below, with the following terms and conditions. '''
len_txt=len(txt)
# date patern like "Month Day, Year"
for m in re.finditer(r'\b\w{3,10}\b \d{1,2}, \d{4,4}', txt):
    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
    max(m.start()-100,0)
    abstract=txt[max(m.start()-100,0): min(m.end()+100,len_txt)]
    print(f'Abstract:\n {abstract}\n')

# company name in single quotes after word between
for m in re.finditer(r'\bbetween\b [\'][A-Za-z\s\.\&\)\(]+[\'] \band\b [\'][A-Za-z\s\.\&\)\(]+[\'] ', txt):
    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
a = re.search(r'\b(and)\b', m.group(0))

conpany_name1=(m.group(0)[:a.start()].split(' ', 1)[1])
conpany_name2=(m.group(0)[a.start():].split(' ', 1)[1])
print ("Company 1: ", conpany_name1)
print ("Company 2: ", conpany_name2)

#match new line //  Blank line
a = re.search(r'(\n|\r)', txt)
print(a)

'''-------------------------------------------------------'''

'''-------------------------------------------------------'''
#solution NLTK

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        #if type(subtree) == Tree and subtree.label() == label:
        if type(subtree) == Tree :
            l=subtree.label()
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append((l,named_entity))
                current_chunk = []
        else:
            continue

    return continuous_chunk

for entity in get_continuous_chunks(txt):
        print(entity[0],  '----- ', entity[1])
        



'''-------------------------------------------------------'''
# https://medium.com/sicara/train-ner-model-with-nltk-stanford-tagger-english-french-german-6d90573a9486
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Java\\bin\\'

# print (os.getenv("JAVAHOME"))
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
locat='C:\\a_machine\\stanford-ner-4.0.0'
st = StanfordNERTagger(f'{locat}\\classifiers\\english.all.3class.distsim.crf.ser.gz',
					   f'{locat}\\stanford-ner.jar',
					   encoding='utf-8')

tokenized_text = word_tokenize(txt)
classified_text = st.tag(tokenized_text)

print(classified_text)

from itertools import groupby
for tag, chunk in groupby(classified_text, lambda x:x[1]):
    if tag != "O":
        print(f'{tag} ---- {" ".join(w for w, t in chunk)}')


'''-------------------------------------------------------'''
#pip install spacy
#64bit environment

import nltk
#pretrained models
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()



doc = nlp(txt)
    
    
for entity in doc.ents:
        print(entity.label_,  '----- ', entity.text)
        

# print([(X.text, X.label_) for X in doc.ents])

# displacy.serve(doc, style="ent")

'''-------------------------------------------------------'''

#NLTK 

# from nltk import word_tokenize, pos_tag
# from nltk import ne_chunk
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# ne_txt_token=word_tokenize(txt)
# ne_txt_tag=pos_tag(ne_txt_token)
# ne_txt_chunk=ne_chunk(ne_txt_tag)
# #split to sentences
# t1=nltk.sent_tokenize(txt)
# tagged_sent = nltk.pos_tag(word_tokenize(t1[0]))
# NE=ne_chunk(tagged_sent)

#NE.draw()