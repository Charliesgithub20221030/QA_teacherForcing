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
post_file=open('QA dataset/stc_weibo_train_post','r')
for i in post_file.readlines():
    post.append(i.split())

post_file=open('QA dataset/stc_weibo_train_response','r')
for i in post_file.readlines():
    response.append(i.split())


# In[3]:


print(len(post),len(response))


# In[12]:


token_stream = []
que_pad=20
ans_pad=10

pair_train1=[]
pair_train2=[]
greedy_search_stop=[]
check_stop=[]
mod_check_stop=[]
count=0
stop_count=0
for key,i in enumerate(post):
    if len(pair_train1)>=10000:
        break
    if len(i)>=ans_pad or len(response[key])>=ans_pad or len(i)<3 or len(response[key])<3:
        continue
    i.append('EOS')
    response[key].append('EOS')
    
    while len(i)<que_pad:
        i.append('PAD')
    while len(response[key])<ans_pad:
        response[key].append('PAD')    
    token_stream.extend(i)
    pair_train1.append(i)    
    token_stream.extend(response[key])
    pair_train2.append(response[key])


# In[13]:


print('num_of_pairs',len(pair_train1))
pair=len(pair_train1)
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

predict_pair=ans_pad


# In[14]:


for i in range(len(pair_train1)):
    for j in range(que_pad):
        pair_train1[i][j]=word2idx[pair_train1[i][j]]

for i in range(len(pair_train2)):
    for j in range(ans_pad):
        pair_train2[i][j]=word2idx[pair_train2[i][j]]


# In[15]:


train_x=[]
train_y=[]
pad_sequence=[word2idx['PAD']]*ans_pad
for i in range(len(pair_train1)):
    for j in range(ans_pad):
        forward=pair_train1[i][:ans_pad]
        backward=pad_sequence[j:ans_pad]
        #print(backward)
        train_x.append(forward+backward+pair_train2[i][:j]) 
        train_y.append([pair_train2[i][j]])

 
train_x=np.array(train_x)
train_y=np.array(train_y)


# In[16]:


print(train_x[:10],train_y[:10])


# In[17]:


print(train_x.shape)
print(train_y.shape)
def get_model():
    dim=256
    inputs = Input(shape=(que_pad,))
    g_emb=Embedding(num_words+1,dim,mask_zero=True, input_length=(que_pad))(inputs)
    decoder = GRU(dim)(g_emb)
    decoder = Dense(num_words,activation="softmax")(decoder)
    model = Model(inputs=inputs , outputs=decoder)
    return model


# In[21]:


# def ppx(y_true, y_pred):
#      loss = K.sparse_categorical_crossentropy(y_true, y_pred)
#      perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
#      return perplexity
   

g_model=get_model()
#sampling_model= multi_gpu_model(get_model(), gpus=2)
g_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='loss', patience=2, verbose=2, mode='auto')


# In[24]:


g_model.fit(train_x,train_y, epochs=49, batch_size=512,validation_split=0.2,verbose=1,callbacks=[earlyStopping])


# In[27]:


def output_sequence(pair_train1,pair_train2,num,g_model):
    word2=[]
    test=[pair_train1[num]]
    #print(test)
    test=np.array(test)
    index=g_model.predict(test)
    index=np.argmax(index[0],axis=-1)      
    word2.append(index)
    for i in range(ans_pad-1):
        test=np.delete(test,ans_pad,1)
        test=np.concatenate([test,[[index]]],axis=1)
        index=g_model.predict(test)
        index=np.argmax(index[0],axis=-1)
        word2.append(index)
        if str(index) == str(word2idx['EOS']):
              break
    que=[]
    sample=[]
    test=[pair_train1[num]+pair_train2[num]]
    for g in test[0]:
          for value, age in word2idx.items():
                if age == g:
                    que.append(value)
    for g in word2:
          for value, age in word2idx.items():
                if age == g:
                    sample.append(value)
    print('question')
    print(''.join(que))
    print('ans_model') 
    print(''.join(sample))
    que=que[0:20]+['   ans:   ']+que[ans_pad:]      
    return  ''.join(que),''.join(sample)
update_g=len(pair_train1)
for i in range(5):
    output_sequence(pair_train1,pair_train2,random.randint(0,random.randint(0,pair-1)),g_model)

old_result=0


# In[ ]:




