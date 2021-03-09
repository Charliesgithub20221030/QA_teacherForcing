#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:04:31 2017

@author: gama
"""
from  keras.utils import multi_gpu_model
import tensorflow as tf
import re
import numpy as np
from keras.models import Model
from keras.layers import Embedding,Masking
from keras.layers import Input, Dense,Reshape,concatenate,Flatten,Activation,Permute,multiply
from keras.layers import GRU,Conv1D,GlobalMaxPooling1D,TimeDistributed,RepeatVector,LSTM,MaxPooling1D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Lambda,Dropout
from keras.utils import to_categorical,multi_gpu_model
import gc
import random
import nltk
import math
from tqdm import tqdm,tqdm_notebook
import csv
import keras.backend as K
from keras.models import load_model
import math


# In[2]:


post=[]
response=[]
post_file=open('weibo QA/stc_weibo_train_post','r')
for i in post_file.readlines():
    post.append(i.split())

post_file=open('weibo QA/stc_weibo_train_response','r')
for i in post_file.readlines():
    response.append(i.split())


# In[3]:


print(len(post),len(response))


# In[103]:


token_stream = [] #control sequence length
que_pad=10
ans_pad=10

pair_trainx1=[]
pair_trainx2=[]
pair_trainy=[]
greedy_search_stop=[]
check_stop=[]
mod_check_stop=[]
count=0
stop_count=0
for key,i in enumerate(post):
    if len(pair_trainx1)>=10000: #select 10000 train set
        break
    if len(i)>=ans_pad-2 or len(response[key])>=ans_pad-2 or len(i)<3 or len(response[key])<3: #restrict train set
        continue
    tempx1=[]
    tempx2=[]
    tempy=[] 
    tempx1.extend(i)
    tempx1.append('EOS')
    
    tempx2.append('BOS')
    tempx2.extend(response[key])
    tempx2.append('EOS')
    
    tempy.extend(response[key])
    tempy.append('EOS')
    
    while len(tempx1)<que_pad:
        tempx1.append('PAD')
    while len(tempx2)<ans_pad:
        tempx2.append('PAD')
    while len(tempy)<ans_pad:
        tempy.append('PAD')        
    token_stream.extend(tempx1)
    token_stream.extend(tempx2)
    token_stream.extend(tempy)
    pair_trainx1.append(tempx1)
    pair_trainx2.append(tempx2)
    pair_trainy.append(tempy)


# In[104]:


pair_trainx1[:3]


# In[105]:


pair_trainx2[:3] #notice the BOS


# In[106]:


pair_trainy[:3] #notice this array and the pair_trainx2


# In[107]:


#TOP=['PAD','EOS']             
#TOP.extend(token_stream)
words=list(set(token_stream))
words.remove('PAD')
#del token_stream

word2idx={}
word2idx['PAD']=0
for i, word in enumerate(words):
    word2idx[word]=i+1
num_words = len(word2idx)
print("num_words:")
print(num_words)
                    
print('process_data')


# In[108]:


word2idx['PAD']


# In[109]:


encoder=[]
decoder=[]
teacher=[]


# In[110]:


for i in pair_trainx1:#preprocess word to dict
    temp=[]
    for j in i:
        temp.append(word2idx[j])
    encoder.append(temp)     


# In[111]:


for i in pair_trainx2:#preprocess word to dict
    temp=[]
    for j in i:
        temp.append(word2idx[j])
    decoder.append(temp) 


# In[112]:


for i in pair_trainy:#preprocess word to dict
    temp=[]
    for j in i:
        temp.append(word2idx[j])
    teacher.append(temp)


# In[113]:


dim=128
input1 = Input(shape=(que_pad,))
input2 = Input(shape=(que_pad,))
emb=Embedding(num_words+1,dim)
context1=emb(input1)
context2=emb(input2)
encoder_outputs,c,h =LSTM(dim,return_state=True)(context1)
encoder_states=[c,h]
decoder_LSTM = LSTM(dim,return_sequences=True)(context2,initial_state=encoder_states)
predict = Dense(num_words+1,activation="softmax")(decoder_LSTM)
model = Model(inputs=[input1,input2], outputs=predict)


# In[114]:


model.summary()


# In[115]:


encoder=np.array(encoder)
decoder=np.array(decoder)
teacher=np.array(teacher)


# In[116]:


#sampling_model= multi_gpu_model(get_model(), gpus=2)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='loss', patience=2, verbose=2, mode='auto')


# In[124]:


model.fit([encoder,decoder],teacher, epochs=100, batch_size=256,verbose=2)


# In[134]:


randindex=random.randint(0,len(pair_trainx1)) #random choose context
print(randindex)
context_test=[]
temp=[]
for j in pair_trainx1[randindex]:
    temp.append(word2idx[j])
context_test.append(temp)
print('q',pair_trainx1[randindex])
print('a',pair_trainy[randindex])
context_test=np.array(context_test)


context_test2=[[0]*que_pad]
context_test2[0][0]=word2idx['BOS'] #First word of the decoder is the BOS 
context_test2=np.array(context_test2)
print('start_array',context_test2)

sequence=[]
for i in range(que_pad-1):
    print(context_test2[0])
    word_distribution=model.predict([context_test,context_test2])
    word=np.argmax(word_distribution[0][i]) #sampling word
    sequence.append(word)
    context_test2[0][i+1]=word #add the sampling word to the next step on decoder input

testdata=[]
for g in sequence:
      for value, transfer_word in word2idx.items():
            if transfer_word == g:
                testdata.append(value)
print(testdata)


# In[ ]:





# In[ ]:




