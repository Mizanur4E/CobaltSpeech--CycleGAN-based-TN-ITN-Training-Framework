#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)


# In[3]:



import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from keras import backend as K

"""##Data importing and preparations"""

def data_processing():
    
    classwise_data = pd.read_csv(r'./classwiseoutputinputv3.csv')
    """## Preparing train & val set"""
    #print(classwise_data)
    from sklearn.model_selection import train_test_split

    (X_date_c)=classwise_data['X_date_c'].tolist()
    (X_letters_c)=classwise_data['X_letters_c'] [:152795].tolist()
    (X_cardinal_c)=classwise_data['X_cardinal_c'] [:133744].tolist()
    (X_verbatim_c)= classwise_data['X_verbatim_c'][:78108].tolist()
    (X_decimal_c)= classwise_data['X_decimal_c'][:9821].tolist()
    (X_measure_c)=classwise_data['X_measure_c'][:14783].tolist()
    (X_money_c)=classwise_data['X_money_c'][:6128].tolist()
    (X_ordinal_c)=classwise_data['X_ordinal_c'] [:12703].tolist()
    (X_time_c)=classwise_data['X_time_c'] [:1465].tolist()
    (X_electronic_c)=classwise_data['X_electronic_c'] [:5162].tolist()
    (X_digit_c)=classwise_data['X_digit_c'] [:5442].tolist()
    (X_fraction_c)=classwise_data['X_fraction_c'] [:1196].tolist()
    (X_telephone_c)=classwise_data['X_telephone_c'][:4024].tolist()
    (X_address_c)=classwise_data['X_address_c'] [:522].tolist()


    (X_date)=classwise_data['X_date'].tolist()
    (X_letters)=classwise_data['X_letters'] [:152795].tolist()
    (X_cardinal)=classwise_data['X_cardinal'] [:133744].tolist()
    (X_verbatim)= classwise_data['X_verbatim'][:78108].tolist()
    (X_decimal)= classwise_data['X_decimal'][:9821].tolist()
    (X_measure)=classwise_data['X_measure'][:14783].tolist()
    (X_money)=classwise_data['X_money'][:6128].tolist()
    (X_ordinal)=classwise_data['X_ordinal'] [:12703].tolist()
    (X_time)=classwise_data['X_time'] [:1465].tolist()
    (X_electronic)=classwise_data['X_electronic'] [:5162].tolist()
    (X_digit)=classwise_data['X_digit'] [:5442].tolist()
    (X_fraction)=classwise_data['X_fraction'] [:1196].tolist()
    (X_telephone)=classwise_data['X_telephone'][:4024].tolist()
    (X_address)=classwise_data['X_address'] [:522].tolist()



    (y_date)=classwise_data['y_date'].tolist() 
    (y_letters)=classwise_data['y_letters'] [:152795].tolist()
    (y_cardinal)=classwise_data['y_cardinal']  [:133744].tolist()
    (y_verbatim)=classwise_data['y_verbatim'][:78108].tolist()
    (y_decimal)=classwise_data['y_decimal'][:9821].tolist()
    (y_measure)=classwise_data['y_measure'][:14783].tolist()
    (y_money)=classwise_data['y_money'] [:6128].tolist()
    (y_ordinal)=classwise_data['y_ordinal']  [:12703].tolist()
    (y_time)=classwise_data['y_time'] [:1465].tolist()
    (y_electronic)=classwise_data['y_electronic']  [:5162].tolist()
    (y_digit)=classwise_data['y_digit']  [:5442].tolist()
    (y_fraction)=classwise_data['y_fraction'][:1196].tolist()
    (y_telephone)=classwise_data['y_telephone'] [:4024].tolist()
    (y_address)=classwise_data['y_address'] [:522].tolist()


    # In[3]:




    X_train_date_c, X_val_date_c, X_train_date, X_val_date, y_train_date, y_val_date = train_test_split(X_date_c,X_date, y_date, test_size=0.015, random_state=42)
    X_train_letters_c, X_val_letters_c, X_train_letters, X_val_letters, y_train_letters, y_val_letters = train_test_split(X_letters_c,X_letters, y_letters, test_size=0.015, random_state=42)
    X_train_cardinal_c, X_val_cardinal_c,X_train_cardinal, X_val_cardinal, y_train_cardinal, y_val_cardinal = train_test_split(X_cardinal_c,X_cardinal, y_cardinal, test_size=0.015, random_state=42)
    X_train_verbatim_c, X_val_verbatim_c,X_train_verbatim, X_val_verbatim, y_train_verbatim, y_val_verbatim = train_test_split(X_verbatim_c,X_verbatim, y_verbatim, test_size=0.015, random_state=42)
    X_train_decimal_c, X_val_decimal_c,X_train_decimal, X_val_decimal, y_train_decimal, y_val_decimal = train_test_split(X_decimal_c,X_decimal, y_decimal, test_size=0.015, random_state=42)
    X_train_measure_c, X_val_measure_c,X_train_measure, X_val_measure, y_train_measure, y_val_measure = train_test_split(X_measure_c,X_measure, y_measure, test_size=0.015, random_state=42)
    X_train_money_c, X_val_money_c,X_train_money, X_val_money, y_train_money, y_val_money = train_test_split(X_money_c,X_money, y_money, test_size=0.015, random_state=42)
    X_train_ordinal_c, X_val_ordinal_c,X_train_ordinal, X_val_ordinal, y_train_ordinal, y_val_ordinal = train_test_split(X_ordinal_c, X_ordinal, y_ordinal, test_size=0.015, random_state=42)
    X_train_time_c, X_val_time_c,X_train_time, X_val_time, y_train_time, y_val_time = train_test_split(X_time_c, X_time, y_time, test_size=0.015, random_state=42)
    X_train_electronic_c, X_val_electronic_c,X_train_electronic, X_val_electronic, y_train_electronic, y_val_electronic = train_test_split(X_electronic_c, X_electronic, y_electronic, test_size=0.015, random_state=42)
    X_train_digit_c, X_val_digit_c,X_train_digit, X_val_digit, y_train_digit, y_val_digit = train_test_split(X_digit_c,X_digit, y_digit, test_size=0.015, random_state=42)
    X_train_fraction_c, X_val_fraction_c, X_train_fraction, X_val_fraction, y_train_fraction, y_val_fraction = train_test_split(X_fraction_c, X_fraction, y_fraction, test_size=0.015, random_state=42)
    X_train_telephone_c, X_val_telephone_c,X_train_telephone, X_val_telephone, y_train_telephone, y_val_telephone = train_test_split(X_telephone_c,X_telephone, y_telephone, test_size=0.015, random_state=42)
    X_train_address_c, X_val_address_c,X_train_address, X_val_address, y_train_address, y_val_address = train_test_split(X_address_c, X_address, y_address, test_size=0.015, random_state=42)


    X_train_c =X_train_date_c+X_train_letters_c+X_train_cardinal_c+X_train_verbatim_c+X_train_decimal_c+X_train_measure_c+X_train_money_c+X_train_ordinal_c+X_train_time_c+X_train_electronic_c+X_train_digit_c+ X_train_fraction_c+X_train_telephone_c+X_train_address_c
    X_train =X_train_date+X_train_letters+X_train_cardinal+X_train_verbatim+X_train_decimal+X_train_measure+X_train_money+X_train_ordinal+X_train_time+X_train_electronic+X_train_digit+ X_train_fraction+X_train_telephone+X_train_address
    y_train= y_train_date+y_train_letters+y_train_cardinal+y_train_verbatim+y_train_decimal+y_train_measure+y_train_money+y_train_ordinal+y_train_time+y_train_electronic+y_train_digit+y_train_fraction+y_train_telephone+y_train_address


    X_val_c =X_val_date_c+X_val_letters_c+X_val_cardinal_c+X_val_verbatim_c+X_val_decimal_c+X_val_measure_c+X_val_money_c+X_val_ordinal_c+X_val_time_c+X_val_electronic_c+X_val_digit_c+ X_val_fraction_c+X_val_telephone_c+X_val_address_c
    X_val =X_val_date+X_val_letters+X_val_cardinal+X_val_verbatim+X_val_decimal+X_val_measure+X_val_money+X_val_ordinal+X_val_time+X_val_electronic+X_val_digit+ X_val_fraction+X_val_telephone+X_val_address
    y_val= y_val_date+y_val_letters+y_val_cardinal+y_val_verbatim+y_val_decimal+y_val_measure+y_val_money+y_val_ordinal+y_val_time+y_val_electronic+y_val_digit+y_val_fraction+y_val_telephone+y_val_address


   


    print(X_val[:10])
    for i in range(len(X_train)):
        X_train[i] = X_train[i].strip()
        y_train[i] = y_train[i].strip()
        X_train_c[i]= X_train_c[i].strip()



    for i in range(len(X_val)):
        X_val[i] = X_val[i].strip()
        y_val[i] = y_val[i].strip()
        X_val_c[i]= X_val_c[i].strip()
    print(X_val[:10])


    # In[4]:




    X_train_c,X_test_c,X_train,X_test,y_train,y_test = train_test_split( X_train_c,X_train, y_train,test_size=0.0015)

    print(len(X_train_c))
    datasetLength =len(X_train_c)#ma data sie 673000

    X_nnrmlzd_c= X_train_c[0:datasetLength]
    X_nnrmlzd= X_train[0:datasetLength]
    X_nrmlzd = y_train[0:datasetLength]



    X_train_c,_,X_train,_,y_train, _ = train_test_split( X_nnrmlzd_c,X_nnrmlzd, X_nrmlzd,test_size=0.0000000001) #shuffles all data

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_nnrmlzd =X_train
    X_nrmlzd =y_train


    pad_size=1
    maxlen=6


    num_samples= len(X_nnrmlzd) 
    gc.collect()
    
    X_data= np.concatenate((X_nnrmlzd,X_nrmlzd))



    return X_data


if __name__ == "__main__":

	X= data_processing()
	corpus = []
	for i in range(len(X)):
    		corpus.append(X[i].split())


	import gensim
	from gensim.models import Word2Vec
	model = Word2Vec(corpus, min_count=1,workers=4,vector_size = 100)
	model.save(word2vec.model")
	sims = model.wv.most_similar('20', topn=1) 
	print(sims)

