#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import gc





classwise_data = pd.read_csv(r'./classwiseoutputinputv3.csv',dtype = 'unicode')


# In[2]:




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


# In[3]:



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

print(len(X_test_c))
datasetLength = 400000          #ma data sie 673000

X_nnrmlzd_c= X_train_c[0:datasetLength]
X_nnrmlzd= X_train[0:datasetLength]
X_nrmlzd = y_train[0:datasetLength]



X_train_c,_,X_train,_,y_train, _ = train_test_split( X_nnrmlzd_c,X_nnrmlzd, X_nrmlzd,test_size=0.0000000001) #shuffles all data

X_nnrmlzd_c=X_train_c 
X_nnrmlzd =X_train
X_nrmlzd =y_train

print(X_nnrmlzd_c[1:4])
print(X_nnrmlzd[1:4])
print(X_nrmlzd[1:4])
print(len(X_nnrmlzd_c),len(X_nnrmlzd),len(X_nrmlzd))

pad_size=1
maxlen1 = 6
maxlen2 = 10  #10 represents number of word generated from one not normalized word


# In[ ]:




"""### Tokenize text and padding"""

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax

tokenizer1 = Tokenizer(num_words=datasetLength )
tokenizer2 = Tokenizer(num_words=datasetLength )
tokenizer1.fit_on_texts(X_nnrmlzd_c)
tokenizer2.fit_on_texts(X_nrmlzd)


X_nnrmlzd_c = tokenizer1.texts_to_sequences(X_nnrmlzd_c)
X_nnrmlzd = tokenizer1.texts_to_sequences(X_nnrmlzd)
X_nrmlzd =  tokenizer2.texts_to_sequences(X_nrmlzd)

vocab_size1 = len(tokenizer1.word_index) + 1  # Adding 1 because of reserved 0 index
vocab_size2 = len(tokenizer2.word_index) + 1

X_nnrmlzd_c = pad_sequences(X_nnrmlzd_c, padding='post', maxlen=maxlen1)
X_nnrmlzd = pad_sequences(X_nnrmlzd, padding='post', maxlen=maxlen1)
X_nrmlzd = pad_sequences(X_nrmlzd, padding='post', maxlen=maxlen2)




token1= tokenizer1.to_json()
token2 = tokenizer2.to_json()

with open('tokenizer1V2.txt', 'w') as f:
  f.write(str(token1))

with open('tokenizer2V2.txt', 'w') as f:
  f.write(str(token2))

#print(vocab_size1,vocab_size2)

#print(tokenizer1.index_word)

print(X_nnrmlzd_c[:5])
print(X_nnrmlzd[:5])
print(X_nrmlzd[:5])


# In[ ]:



X_train_c=(X_nnrmlzd_c)
X_train=(X_nnrmlzd)
y_train=(X_nrmlzd)


# In[ ]:



# Check keras version
import tensorflow as tf
import os 
os.environ['TF_CUDNN_DETERMINISTIC']='1'


import keras
if keras.__version__ < '2.3.1':
	print('Use Keras 2.3.1 or later')
	exit(1)
# If multiple copies of the OpenMP runtime is available,
# setting 'KMP_DUPLICATE_LIB_OK=TRUE' allows the program to execute
# I am facing this issue after I have upgraded to Tensorflow 2.
from os import environ
environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Lambda
from keras import backend as K
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Flatten, Maximum
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.models import Input
from tensorflow.keras.optimizers import Adam
from random import random
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import argmax
from numpy import array
from numpy.random import randint
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy.random import shuffle


# In[ ]:


#Scoring Fucntions
#functions for generating BLEU, ACCURACY and WER 
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from jiwer import wer 
from sklearn.metrics import accuracy_score

def average_bleu(refa, hypa):
    
  smoothie = SmoothingFunction().method2
  n= len(refa)
  net =0
  for i in range(n):
    
    a= sentence_bleu([refa[i]],hypa[i], smoothing_function=smoothie)
    net= net + a
  me = net/n
  return me

def acc_scorer(y_pred,y_true):
    a= 0
    for i in range(len(y_true)):
        a+= accuracy_score(y_true[i],y_pred[i])
    a= a/len(y_true)
    return a
def accuracy_printerXY(X,y_true,gen):
    inp= tokenizer1.texts_to_sequences(X)
    inp =pad_sequences(inp, padding='post', maxlen=maxlen1)
    out = gen.predict(inp)
    out = np.argmax(out,axis=-1)
    text_out= tokenizer2.sequences_to_texts(out)
    ground_truth =y_true

    ground_truth_seq = tokenizer2.texts_to_sequences(ground_truth)
    ground_truth_seq =pad_sequences(ground_truth_seq, padding='post', maxlen=maxlen2)
    text_out= np.asarray(text_out)
    ground_truth= ground_truth
    text_out= text_out.tolist()
    
    acc= acc_scorer(out,ground_truth_seq)
    '''
    print(text_out[:10])
    print('------------------------------')
    print(ground_truth[:10])'''
    w= wer(ground_truth,text_out)
    #print('WER is:', w*100)

    #print('Accuracy is:',acc)
    smoothie = SmoothingFunction().method4
    score = average_bleu(ground_truth,text_out)
    #print('Average BLEU score is: ',score)
    #print('\n')
    
    return w,acc,score


def accuracy_printerYX(X,y_true,gen):

    wer_avg= 0
    acc_avg = 0
    bleu_avg =0 

    n_s = len(X)

    n_s /=20
    n_s +=1

    for i in range(int(n_s)):
      sample = X[20*i:20*(i+1)]
      
      if sample == []: 
        continue
      inp= tokenizer2.texts_to_sequences(sample)
      inp =pad_sequences(inp, padding='post', maxlen=maxlen2)
      out = gen.predict(inp)
      out = np.argmax(out,axis=-1)
      text_out= tokenizer1.sequences_to_texts(out)
      ground_truth =y_true[20*i:20*(i+1)]

      ground_truth_seq = tokenizer2.texts_to_sequences(ground_truth)
      ground_truth_seq =pad_sequences(ground_truth_seq, padding='post', maxlen=maxlen1)
      text_out= np.asarray(text_out)
      ground_truth= ground_truth
      text_out= text_out.tolist()
      
      acc= acc_scorer(out,ground_truth_seq)
      ''' 
      print(text_out[:10])
      print('------------------------------')
      print(ground_truth[:10]) '''

      w= wer(ground_truth,text_out)
      #print('WER is:', w*100)

      #print('Accuracy is:',acc)
      smoothie = SmoothingFunction().method4
      score = average_bleu(ground_truth,text_out)
      #print('Average BLEU score is: ',score)
      #print('\n')

      wer_avg += w*len(sample)
      acc_avg += acc*len(sample)
      bleu_avg += score*len(sample)
    

    wer_avg /= len(X)
    acc_avg /= len(X)
    bleu_avg /= len(X)

    return wer_avg,acc_avg,bleu_avg


# In[ ]:


def performanceXY(model):

  edges = [3876 , 2292 , 2007 , 1172 , 148 , 222 , 92 , 191 , 22 , 78 , 82 , 18 , 61 , 8]
  classes =['date','letters','cardinal','verbatim','decimal','measure','money','ordinal','time',
            'electronic','digit', 'fraction','telephone','address']

  terminal=0
  initial =0

  wer_classwise = []
  acc_classwise = []
  bleu_classwise = []
  for i in range(14):
      
      terminal= initial + edges[i]
      #print('For class: ',classes[i])
      wr, acc, bleu = accuracy_printerXY(X_val_c[initial:terminal],y_val[initial:terminal],model)
      wer_classwise.append(wr)
      acc_classwise.append(acc)
      bleu_classwise.append(bleu)
      
      initial = terminal
      
      


  print('WER classwise:',wer_classwise)
  print('Accuracy classwise:',acc_classwise)
  print('BLEU classwise', bleu_classwise)

  
  print('Overall scores:')
  print(accuracy_printerXY(X_val_c,y_val,model))


def performanceYX(model):

  edges = [3876 , 2292 , 2007 , 1172 , 148 , 222 , 92 , 191 , 22 , 78 , 82 , 18 , 61 , 8]
  classes =['date','letters','cardinal','verbatim','decimal','measure','money','ordinal','time',
            'electronic','digit', 'fraction','telephone','address']

  terminal=0
  initial =0

  wer_classwise = []
  acc_classwise = []
  bleu_classwise = []
  for i in range(14):
      
      terminal= initial + edges[i]
      #print('For class: ',classes[i])
      wr, acc, bleu = accuracy_printerYX(y_val[initial:terminal],X_val[initial:terminal],model)
      wer_classwise.append(wr)
      acc_classwise.append(acc)
      bleu_classwise.append(bleu)
      
      initial = terminal
      

  print('WER classwise:',wer_classwise)
  print('Accuracy classwise:',acc_classwise)
  print('BLEU classwise', bleu_classwise)

  print('Overall scores:')
  print(accuracy_printerYX(y_val,X_val,model))


# In[ ]:



"""## Functions for generating training data samples"""

"all set to go"
def generate_real_samples2(dataset_c,dataset, n_samples, out_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# extract selected sequences
	X_c = dataset_c[ix]
    
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, out_shape))
	return X_c,X, y

# select a batch of random samples, returns real dequences and target labels
def generate_real_samples(dataset, n_samples, out_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# extract selected sequences
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, out_shape))
	return X, y

# generate a batch of fake sequences (produced by the generators) and taget labels
def generate_fake_samples(g_model, dataset, out_shape):
	# generate fake instance
	X_sftmax = g_model.predict(dataset)
	X_int = list()
	for mat in X_sftmax:
		vec = [argmax(i) for i in mat]
		X_int.append(vec)
	X = array(X_int)
	# create 'fake' class labels (0)
	y = zeros((len(X), out_shape))
	return X, y


# update data pool for fake sequences
def update_data_pool(pool, data, max_size=500):
	selected = list()
	for seq in data:
		if len(pool) < max_size:
			# stock the pool
			pool.append(seq)
			selected.append(seq)
		elif random() < 0.5:
			# use data, but don't add it to the pool
			selected.append(seq)
		else:
			# replace an existing seq and use replaced seq
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = seq
	return asarray(selected)

# select a batch of random samples for seq2seq training
def generate_real_samples_seq(trainX_c,trainX, trainY, n_samples):
	# choose random instances
	ix = randint(0, trainX.shape[0], n_samples)
	# retrieve selected images
	X_c = trainX_c[ix]
	X = trainX[ix]
	Y = trainY[ix]
	return X_c,X, Y

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

"all set to go"

'''
# define generator model
def define_generator(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	# summarize defined model

	return model
'''


"all set to go"


# define generator model
def define_generator(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(Bidirectional(LSTM(n_units)))
	model.add(RepeatVector(tar_timesteps))
	model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy',metrics= ['accuracy'])
	# summarize defined model

	return model


# define discriminator model
def define_discriminator(vocab_size, max_length):
	model = Sequential()
	model.add(Embedding(vocab_size, 256, input_length=max_length))
	model.add(Conv1D(filters=64, kernel_size= 3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile network
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model

	return model

"all set to go"

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, timestep1, timestep2):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input1 = Input(shape=(timestep1,))
	gen1_out_sftmax = g_model_1(input1)
	ArgmaxLayer = Lambda(lambda x: K.cast(K.argmax(x), dtype='float32'))
	gen1_out = ArgmaxLayer(gen1_out_sftmax)
	output_d = d_model(gen1_out)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	input2 = Input(shape=(timestep2,))
	gen2_out_sftmax = g_model_2(input2)
	gen2_out = ArgmaxLayer(gen2_out_sftmax)
	output_b= g_model_1(gen2_out)
	# define model graph
	model = Model([input1, input2], [output_d, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.2)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[2.0, 5.0, 5.0], optimizer=opt)

	return model


# In[ ]:

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


"""## Function for training cyclegan models"""



def train(d_model_X, d_model_Y, g_model_XtoY, g_model_YtoX, c_model_XtoY, c_model_YtoX, dataset, vocabSize, tokenizer): 
    
    # define properties of the training run
    n_epochs, n_batch, = 13, 128
    # determine the output shape of the discriminator
    n_out = d_model_X.output_shape[1]
    # unpack dataset and train-test split
    trainX_c, trainX, trainY = dataset
    

    vocabSizeX, vocabSizeY = vocabSize
    tokenizerX ,tokenizerY = tokenizer
    # prepare data pool for fakes
    poolX, poolY = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainX) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    gX_loss = list()
    gY_loss = list()
    
    for i in range(n_steps):
        # select a batch of real samples
        X_realX_c,X_realX, y_realX = generate_real_samples2(trainX_c,trainX, n_batch, n_out)
        X_realY, y_realY = generate_real_samples(trainY, n_batch, n_out)
        X_real_trainX_c,X_real_trainX, X_real_trainY = generate_real_samples_seq(trainX_c,trainX, trainY, n_batch)
        # generate a batch of fake samples
        X_fakeX, y_fakeX = generate_fake_samples(g_model_YtoX, X_realY, n_out)
        X_fakeY, y_fakeY = generate_fake_samples(g_model_XtoY, X_realX_c, n_out)    #think again
        # update fakes from pool
        X_fakeX = update_data_pool(poolX, X_fakeX)
        X_fakeY = update_data_pool(poolY, X_fakeY)
        # convert integer output to softmax
        X_real_trainX_sftmax = encode_output(X_real_trainX, vocabSizeX)
        X_real_trainY_sftmax = encode_output(X_real_trainY, vocabSizeY)
        X_realX_sftmax = encode_output(X_realX, vocabSizeX)
        X_realY_sftmax = encode_output(X_realY, vocabSizeY)
        # update generator Y->X via seq2seq loss
        gy_loss, _ = g_model_YtoX.train_on_batch(X_real_trainY, X_real_trainX_sftmax)  
        # update generator Y->X via adversarial and cycle loss
        g_loss2, _, _, _  = c_model_YtoX.train_on_batch([X_realY, X_realX_c], [y_realX, X_realY_sftmax, X_realX_sftmax]) #see here
        # update discriminator for X -> [real/fake]
        dX_loss1, _ = d_model_X.train_on_batch(X_realX, y_realX)
        dX_loss2, _ = d_model_X.train_on_batch(X_fakeX, y_fakeX)
        # update generator X->Y via seq2seq loss
        gx_loss, _ = g_model_XtoY.train_on_batch(X_real_trainX_c, X_real_trainY_sftmax)
        # update generator X->Y via adversarial and cycle loss
        g_loss1, _, _, _  = c_model_XtoY.train_on_batch([X_realX_c, X_realY], [y_realY, X_realX_sftmax, X_realY_sftmax])
        # update discriminator for Y -> [real/fake]
        dY_loss1, _ = d_model_Y.train_on_batch(X_realY, y_realY)
        dY_loss2, _ = d_model_Y.train_on_batch(X_fakeY, y_fakeY)
        # summarize performance
        
        #tf.keras.backend.clear_session()
        if (i+1)%20 == 0:
            print('>[%d] dX[%.3f,%.3f] dY[%.3f,%.3f] g[%.3f,%.3f] gX[%.3f] gY[%.3f]' % (i+1, dX_loss1,dX_loss2, dY_loss1,dY_loss2, g_loss1,g_loss2,gx_loss,gy_loss))
            gc.collect()
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            print('Summary of X-> Y conversion:')
            performanceXY(g_model_XtoY)
            print('Summary of Y-> X conversion:')
            performanceYX(g_model_YtoX)

         
            gX_loss.append(gx_loss)
            gY_loss.append(gy_loss)
            #g_model_XtoY.save('modelXY_V2')
            #g_model_YtoX.save('modelYX_V2')
            
    print(gX_loss)
    print(gY_loss)


tokenizer =[tokenizer1,tokenizer2]
timestepX = maxlen1
timestepY = maxlen2

"All set to go !!"


if __name__ == "__main__":
	# load dataset

    
	dataX_c = np.array(X_train_c)
	dataX = np.array(X_train)
	dataY =  np.array(y_train)
  

	
	dataset = [dataX_c,dataX, dataY]

	vocabSize =[vocab_size1,vocab_size2]
	
	
	# generator: X -> Y
	g_model_XtoY = define_generator(vocabSize[0], vocabSize[1], timestepX, timestepY,256)
	# generator: Y -> X
	g_model_YtoX = define_generator(vocabSize[1], vocabSize[0], timestepY, timestepX, 256)
	# discriminator: X -> [spoken/written]
	d_model_X = define_discriminator(vocabSize[0], timestepX)
	# discriminator: B -> [spoken/written]
	d_model_Y = define_discriminator(vocabSize[1], timestepY)
	# composite: X -> Y -> [spoken/written, X]
	c_model_XtoY = define_composite_model(g_model_XtoY, d_model_Y, g_model_YtoX, timestepX, timestepY)
	# composite: Y -> X -> [spoken/written, Y]
	c_model_YtoX = define_composite_model(g_model_YtoX, d_model_X, g_model_XtoY, timestepY, timestepX)
	# train models
	train(d_model_X, d_model_Y, g_model_XtoY, g_model_YtoX, c_model_XtoY, c_model_YtoX, dataset, vocabSize, tokenizer)

"""--------------------------------Let's finish it---------------------------------"""

