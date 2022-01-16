'''
trained generator with both seq2seq loss (classical-crossentropy)  and adversarial loss.
Discriminator has normal binary crossentropy loss
'''
from __future__ import print_function, division
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD,  RMSprop
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import array
from numpy import argmax
from keras import backend as K
from random import random



"""##Data importing and preparations"""
def Data_preparator1(datasize, timestep):
    
    classwise_data = pd.read_csv(r'./classwiseoutputinputv3.csv')
    """## Preparing train & val set"""
    #print(classwise_data)
    from sklearn.model_selection import train_test_split
    (X_date_c)=classwise_data['X_date_c'].tolist()
    (X_letters_c)=classwise_data['X_letters_c'] [:152795].tolist()
    (X_cardinal_c)=classwise_data['X_cardinal_c'] [:133744].tolist()
    #(X_verbatim_c)= classwise_data['X_verbatim_c'][:78108].tolist()
    (X_decimal_c)= classwise_data['X_decimal_c'][:9821].tolist()
    (X_measure_c)=classwise_data['X_measure_c'][:14783].tolist()
    #(X_money_c)=classwise_data['X_money_c'][:6128].tolist()
    (X_ordinal_c)=classwise_data['X_ordinal_c'] [:12703].tolist()
    (X_time_c)=classwise_data['X_time_c'] [:1465].tolist()



    (X_date)=classwise_data['X_date'].tolist()
    (X_letters)=classwise_data['X_letters'] [:152795].tolist()
    (X_cardinal)=classwise_data['X_cardinal'] [:133744].tolist()
    #(X_verbatim)= classwise_data['X_verbatim'][:78108].tolist()
    (X_decimal)= classwise_data['X_decimal'][:9821].tolist()
    (X_measure)=classwise_data['X_measure'][:14783].tolist()
    #(X_money)=classwise_data['X_money'][:6128].tolist()
    (X_ordinal)=classwise_data['X_ordinal'] [:12703].tolist()
    (X_time)=classwise_data['X_time'] [:1465].tolist()



    (y_date)=classwise_data['y_date'].tolist() 
    (y_letters)=classwise_data['y_letters'] [:152795].tolist()
    (y_cardinal)=classwise_data['y_cardinal']  [:133744].tolist()
    #(y_verbatim)=classwise_data['y_verbatim'][:78108].tolist()
    (y_decimal)=classwise_data['y_decimal'][:9821].tolist()
    (y_measure)=classwise_data['y_measure'][:14783].tolist()
    #(y_money)=classwise_data['y_money'] [:6128].tolist()
    (y_ordinal)=classwise_data['y_ordinal']  [:12703].tolist()
    (y_time)=classwise_data['y_time'] [:1465].tolist()



    # In[3]:




    X_train_date_c, X_val_date_c, X_train_date, X_val_date, y_train_date, y_val_date = train_test_split(X_date_c,X_date, y_date, test_size=0.015, random_state=42)
    X_train_letters_c, X_val_letters_c, X_train_letters, X_val_letters, y_train_letters, y_val_letters = train_test_split(X_letters_c,X_letters, y_letters, test_size=0.015, random_state=42)
    X_train_cardinal_c, X_val_cardinal_c,X_train_cardinal, X_val_cardinal, y_train_cardinal, y_val_cardinal = train_test_split(X_cardinal_c,X_cardinal, y_cardinal, test_size=0.015, random_state=42)
    #X_train_verbatim_c, X_val_verbatim_c,X_train_verbatim, X_val_verbatim, y_train_verbatim, y_val_verbatim = train_test_split(X_verbatim_c,X_verbatim, y_verbatim, test_size=0.015, random_state=42)
    X_train_decimal_c, X_val_decimal_c,X_train_decimal, X_val_decimal, y_train_decimal, y_val_decimal = train_test_split(X_decimal_c,X_decimal, y_decimal, test_size=0.015, random_state=42)
    X_train_measure_c, X_val_measure_c,X_train_measure, X_val_measure, y_train_measure, y_val_measure = train_test_split(X_measure_c,X_measure, y_measure, test_size=0.015, random_state=42)
    #X_train_money_c, X_val_money_c,X_train_money, X_val_money, y_train_money, y_val_money = train_test_split(X_money_c,X_money, y_money, test_size=0.015, random_state=42)
    X_train_ordinal_c, X_val_ordinal_c,X_train_ordinal, X_val_ordinal, y_train_ordinal, y_val_ordinal = train_test_split(X_ordinal_c, X_ordinal, y_ordinal, test_size=0.015, random_state=42)
    X_train_time_c, X_val_time_c,X_train_time, X_val_time, y_train_time, y_val_time = train_test_split(X_time_c, X_time, y_time, test_size=0.015, random_state=42)
   

    X_train_c =X_train_date_c+X_train_letters_c+X_train_measure_c+X_train_cardinal_c+X_train_decimal_c+X_train_ordinal_c+X_train_time_c
    X_train   =X_train_date+X_train_letters+X_train_measure+X_train_cardinal+X_train_decimal+X_train_ordinal+X_train_time
    y_train=   y_train_date+y_train_letters+y_train_measure+y_train_cardinal+y_train_decimal+y_train_ordinal+y_train_time

    X_val_c=X_val_date_c+X_val_letters_c+X_val_measure_c+X_val_cardinal_c+X_val_decimal_c+X_val_ordinal_c+X_val_time_c
    X_val =X_val_date+X_val_letters+X_val_measure+X_val_cardinal+X_val_decimal+X_val_ordinal+X_val_time
    y_val= y_val_date+y_val_letters+y_val_measure+y_val_cardinal+y_val_decimal+y_val_ordinal+y_val_time


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

    print(len(X_train_c))
    datasetLength = datasize #ma data sie 673000

    X_nnrmlzd_c= X_train_c[0:datasetLength]
    X_nnrmlzd= X_train[0:datasetLength]
    X_nrmlzd = y_train[0:datasetLength]



    X_train_c,_,X_train,_,y_train, _ = train_test_split( X_nnrmlzd_c,X_nnrmlzd, X_nrmlzd,test_size=0.0000000001) #shuffles all data

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_nnrmlzd =X_train
    X_nrmlzd =y_train


    
    
    maxlen= timestep
    
    #split each samples into words
    X= []
    y= []
    
    for i in range(len(X_nnrmlzd)):
        
        X.append(X_nnrmlzd[i].split())
        y.append(X_nrmlzd[i].split())
    
    data =[]
    for s in X:
        data.append(s)
    for s in y:
        data.append(s)
    tokenizer = Tokenizer()    
    tokenizer.fit_on_texts(data)
    
    X_seq = tokenizer.texts_to_sequences(X)
    y_seq = tokenizer.texts_to_sequences(y)
    vocab_size = len(tokenizer.word_index)+1
    
    '''  
    tokenizer1 = Tokenizer()
    tokenizer1.fit_on_texts(X)
    
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(y)
    X_seq = tokenizer1.texts_to_sequences(X)
    y_seq = tokenizer2.texts_to_sequences(y)
    
    vocab_size1 = len(tokenizer1.word_index)+1
    vocab_size2 = len(tokenizer2.word_index)+1
    '''

    X_seq = pad_sequences(X_seq,maxlen = maxlen)
    y_seq = pad_sequences(y_seq,maxlen= maxlen)
    
    X_train, X_test, y_train,y_test = train_test_split(X_seq,y_seq,test_size = 0.02,random_state=42)
    
    X_enc = to_categorical(X_train,num_classes= vocab_size)
    y_enc = to_categorical(y_train,num_classes = vocab_size)
    
    
    X_val = to_categorical(X_test,num_classes= vocab_size)
    y_val = to_categorical(y_test,num_classes = vocab_size)
    
    
    return X_enc,y_enc,[X_val,y_val],tokenizer,vocab_size


#implements wessertein loss and rmsprop as optimizer
class NoEmbeddingGAN():
    
    def __init__(self, vocab_size,tokenizer):
        
        self.n_critic= 5
        self.clip_value = 0.01
        
        self.timestep = 7
        self.embedding_dim = 100
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        #D_optimizer = Adam(0.0002,0.0,0.9)
        D_optimizer = Adam(0.0002,0.5)
        optimizer =Adam(0.05)
        G_optimizer = Adam(0.0002,0.5,0.9)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        #self.discriminator.compile(loss=self.wessertein_loss,optimizer= optimizer, metrics=['accuracy'])
        self.discriminator.compile(loss='binary_crossentropy',optimizer=D_optimizer,metrics=['accuracy'])
        
        inp1 = layers.Input(shape=(self.timestep,self.vocab_size,))
        gen_out = self.generator(inp1)
        self.discriminator.trainable = False
        dis_out = self.discriminator([gen_out,inp1])
        self.gan = Model(inputs= inp1,outputs= dis_out)
        self.gan.compile(loss='binary_crossentropy',optimizer=G_optimizer,metrics='accuracy')
        #self.gan.compile(loss=self.wessertein_loss,optimizer=optimizer,metrics=['accuracy'])
        
        
    def build_generator(self):
        
        inp1= layers.Input(shape=(self.timestep,self.vocab_size,))
        inp= layers.Dense(self.embedding_dim,use_bias=False)(inp1)
        inp = layers.Bidirectional(layers.LSTM(units=256))(inp)
        inp = layers.RepeatVector(self.timestep)(inp)
        inp =layers.Bidirectional(layers.LSTM(units=256,return_sequences= True))(inp)
        inp = layers.TimeDistributed(layers.Dense(self.vocab_size))(inp)
        #inp = layers.Lambda(lambda x: 4*x, dtype='float32')(inp)
        out = layers.Softmax()(inp)
        model = Model(inputs= inp1,outputs=out)
        model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics='accuracy')
        
        return model
    
    def build_discriminator(self):
        
        inp1 = layers.Input(shape=(self.timestep,self.vocab_size,)) #inputs seq
        den_inp1 = layers.Dense(self.embedding_dim,use_bias=False)(inp1) #contains embedding
        
        condtn = layers.Input(shape= (self.timestep,self.vocab_size,))  #condition
        den_condtn= layers.Dense(self.embedding_dim,use_bias=False)(condtn)
        
        emb = layers.concatenate([den_inp1,den_condtn],axis =1)
        #emb = layers.Dense(200,use_bias=False)(merged)
        
        conv1 = layers.Conv1D(128,2,padding='same',activation='tanh')(emb)
        pool1 = layers.MaxPool1D(2)(conv1)
        flat1 = layers.Flatten()(pool1)
        
        conv2 = layers.Conv1D(128,3,padding='same',activation='tanh')(emb)
        pool2 = layers.MaxPool1D(2)(conv2)
        flat2 = layers.Flatten()(pool2)
 
        conv3 = layers.Conv1D(128,4,padding='same',activation='tanh')(emb)
        pool3 = layers.MaxPool1D(2)(conv3)
        flat3 = layers.Flatten()(pool3)
 
        conv  = layers.concatenate([flat1,flat2,flat3],axis =-1)

        den = layers.Dense(50,activation='tanh')(conv)
        #den = layers.Dense(30,activation='relu')(den)
        out = layers.Dense(1,activation='sigmoid')(den)
        
        model = Model(inputs=[inp1,condtn],outputs= out)
        
        return model
  

    def pretrain(self,X_nnrmlzd, X_nrmlzd,epochs):
        
        
        epoch = epochs
        batch_num = int(X_nrmlzd.shape[0]/128)
        
        
        
        print('Pretraining Generator..............')
        #self.generator.fit(X_nnrmlzd,X_nrmlzd,epochs= 1, batch_size = 64,verbose= 1,validation_split=0.1)
        for _ in range(5):
            for j in range(batch_num):
                
                ix = np.random.randint(0,X_nrmlzd.shape[0],128)
                g= self.generator.train_on_batch(X_nnrmlzd[ix],X_nrmlzd[ix])
                
                if j%100 == 0:
                    print('Gen Loss[%.5f],Accuracy[%.2f %%]'%(g[0],100*g[1]))
                if j % 600 == 0 and j != 0:
                    self.generator.evaluate(X_nnrmlzd[:1000],X_nrmlzd[:1000])    
      
        print('Pretraining Discriminator.............')       

        fake_label = np.zeros((128,1))
        valid_label = np.ones((128,1))
        
        for k in range(epoch):
            
            for i in range(batch_num):
                
                ix = np.random.randint(0,X_nrmlzd.shape[0],128)
                x_valid= X_nnrmlzd[ix]
                y_fake = self.generator.predict(X_nnrmlzd[ix])
                y_valid = X_nrmlzd[ix]
                '''  
                y_temp = np.concatenate([y_fake,y_valid])
                x_temp = np.concatenate([x_valid,x_valid])
                
                label = np.concatenate([fake_label,valid_label])
                Y, X_ref, label = shuffle(y_temp,x_temp,label)
                d_loss= self.discriminator.train_on_batch([Y,X_ref], label)
                '''
                
                d_loss_real = self.discriminator.train_on_batch([y_valid,x_valid], valid_label)
                d_loss_fake = self.discriminator.train_on_batch([y_fake,x_valid], fake_label)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                
                #print(d_loss_real)
                if i % 50 == 0:
                    print ("%d %d [D loss: %f, acc.: %.5f%%]" % (k,i, d_loss[0], 100*d_loss[1]))

    def wessertein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def train(self,X_nnrmlzd, X_nrmlzd,val_data,epochs,sample_interval,batch_size= 128):

        X_train= X_nnrmlzd  #embedding_vector
        y_train= X_nrmlzd   #one
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        poolX = []
        

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            

            d_loss= [0.0, 0.0]



            # Generate a half batch of new normalized text
            #noise_1 =np.random.normal(0,1,(half_batch,self.timestep,self.embedding_dim))
            
            for _ in range(self.n_critic):
                                
                ix = np.random.randint(0,X_nrmlzd.shape[0],128)
                x_valid= X_nnrmlzd[ix]
                y_fake = self.generator.predict(X_nnrmlzd[ix])
                y_valid = X_nrmlzd[ix]
                
                '''  
                y_temp = np.concatenate([y_fake,y_valid])
                x_temp = np.concatenate([x_valid,x_valid])
                
                label = np.concatenate([fake,valid])
                Y, X_ref, label = shuffle(y_temp,x_temp,label)
                
                d_loss= self.discriminator.train_on_batch([Y,X_ref], label)
                '''
                
                d_loss_real = self.discriminator.train_on_batch([y_valid,x_valid], valid)
                d_loss_fake = self.discriminator.train_on_batch([y_fake,x_valid], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                '''  
                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                '''


            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            #noise=np.random.normal(0,1,(4*batch_size,self.timestep,self.embedding_dim))
            

            # Train the generator
            g_loss = self.generator.train_on_batch(X_train[idx],y_train[idx])
            gan_loss = self.gan.train_on_batch(X_train[idx], valid)
           
            # Plot the progress
            if epoch % 100 ==0:
               
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%] [GAN loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0],100*g_loss[1],gan_loss[0],100*gan_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                #set test size 25
                
                ix = np.random.randint(0,val_data[0].shape[0],10)
                
                self.generator.evaluate(val_data[0],val_data[1],batch_size=128,verbose=1)
                if  epoch !=0:
                    self.sample_text(val_data[0][ix],val_data[1][ix])

        return 0

        # update data pool for fake sequences
    def update_data_pool(self,pool, data, max_size=500):
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
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = seq
        return np.asarray(selected),pool
    
    def sample_text(self,x,y_true):
        
        #noise=np.random.normal(0,1,(x.shape[0],self.timestep,self.embedding_dim))
        y= self.generator.predict(x)
        y = np.argmax(y,axis = -1)
        y_true= np.argmax(y_true,axis=-1)
        y_true_text = self.tokenizer.sequences_to_texts(y_true)
        y_pred_text = self.tokenizer.sequences_to_texts(y)
        x= np.argmax(x,axis=-1)
        x_text = self.tokenizer.sequences_to_texts(x)
        
        for i in range(len(x_text)):
            print(x_text[i],'--',y_true_text[i],'--',y_pred_text[i])
        
        
        
X_enc,y_enc,val_set,tokenizer,vocab_size= Data_preparator1(200000, 7)
lion1 = NoEmbeddingGAN(vocab_size,tokenizer)
lion1.pretrain(X_enc,y_enc,4)
lion1.train(X_enc,y_enc,val_set,1000000,400)
