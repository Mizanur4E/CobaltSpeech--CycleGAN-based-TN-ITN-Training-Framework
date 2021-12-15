from __future__ import print_function, division
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
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

def Data_preparator(datasize, timestep):
    
    classwise_data = pd.read_csv(r'./classwiseoutputinputv3.csv')
    """## Preparing train & val set"""
    #print(classwise_data)
    from sklearn.model_selection import train_test_split

    (X_date_c)=classwise_data['X_date_c'].tolist()
    (X_letters_c)=classwise_data['X_letters_c'] [:152795].tolist()
    (X_cardinal_c)=classwise_data['X_cardinal_c'] [:133744].tolist()
    #(X_verbatim_c)= classwise_data['X_verbatim_c'][:78108].tolist()
    (X_decimal_c)= classwise_data['X_decimal_c'][:9821].tolist()
    #(X_measure_c)=classwise_data['X_measure_c'][:14783].tolist()
    #(X_money_c)=classwise_data['X_money_c'][:6128].tolist()
    (X_ordinal_c)=classwise_data['X_ordinal_c'] [:12703].tolist()
    (X_time_c)=classwise_data['X_time_c'] [:1465].tolist()



    (X_date)=classwise_data['X_date'].tolist()
    (X_letters)=classwise_data['X_letters'] [:152795].tolist()
    (X_cardinal)=classwise_data['X_cardinal'] [:133744].tolist()
    #(X_verbatim)= classwise_data['X_verbatim'][:78108].tolist()
    (X_decimal)= classwise_data['X_decimal'][:9821].tolist()
    #(X_measure)=classwise_data['X_measure'][:14783].tolist()
    #(X_money)=classwise_data['X_money'][:6128].tolist()
    (X_ordinal)=classwise_data['X_ordinal'] [:12703].tolist()
    (X_time)=classwise_data['X_time'] [:1465].tolist()



    (y_date)=classwise_data['y_date'].tolist() 
    (y_letters)=classwise_data['y_letters'] [:152795].tolist()
    (y_cardinal)=classwise_data['y_cardinal']  [:133744].tolist()
    #(y_verbatim)=classwise_data['y_verbatim'][:78108].tolist()
    (y_decimal)=classwise_data['y_decimal'][:9821].tolist()
    #(y_measure)=classwise_data['y_measure'][:14783].tolist()
    #(y_money)=classwise_data['y_money'] [:6128].tolist()
    (y_ordinal)=classwise_data['y_ordinal']  [:12703].tolist()
    (y_time)=classwise_data['y_time'] [:1465].tolist()



    # In[3]:




    X_train_date_c, X_val_date_c, X_train_date, X_val_date, y_train_date, y_val_date = train_test_split(X_date_c,X_date, y_date, test_size=0.015, random_state=42)
    X_train_letters_c, X_val_letters_c, X_train_letters, X_val_letters, y_train_letters, y_val_letters = train_test_split(X_letters_c,X_letters, y_letters, test_size=0.015, random_state=42)
    X_train_cardinal_c, X_val_cardinal_c,X_train_cardinal, X_val_cardinal, y_train_cardinal, y_val_cardinal = train_test_split(X_cardinal_c,X_cardinal, y_cardinal, test_size=0.015, random_state=42)
    #X_train_verbatim_c, X_val_verbatim_c,X_train_verbatim, X_val_verbatim, y_train_verbatim, y_val_verbatim = train_test_split(X_verbatim_c,X_verbatim, y_verbatim, test_size=0.015, random_state=42)
    X_train_decimal_c, X_val_decimal_c,X_train_decimal, X_val_decimal, y_train_decimal, y_val_decimal = train_test_split(X_decimal_c,X_decimal, y_decimal, test_size=0.015, random_state=42)
    #X_train_measure_c, X_val_measure_c,X_train_measure, X_val_measure, y_train_measure, y_val_measure = train_test_split(X_measure_c,X_measure, y_measure, test_size=0.015, random_state=42)
    #X_train_money_c, X_val_money_c,X_train_money, X_val_money, y_train_money, y_val_money = train_test_split(X_money_c,X_money, y_money, test_size=0.015, random_state=42)
    X_train_ordinal_c, X_val_ordinal_c,X_train_ordinal, X_val_ordinal, y_train_ordinal, y_val_ordinal = train_test_split(X_ordinal_c, X_ordinal, y_ordinal, test_size=0.015, random_state=42)
    X_train_time_c, X_val_time_c,X_train_time, X_val_time, y_train_time, y_val_time = train_test_split(X_time_c, X_time, y_time, test_size=0.015, random_state=42)
   

    X_train_c =X_train_date_c+X_train_letters_c+X_train_cardinal_c+X_train_decimal_c+X_train_ordinal_c+X_train_time_c
    X_train   =X_train_date+X_train_letters+X_train_cardinal+X_train_decimal+X_train_ordinal+X_train_time
    y_train=   y_train_date+y_train_letters+y_train_cardinal+y_train_decimal+y_train_ordinal+y_train_time

    X_val_c =X_val_date_c+X_val_letters_c+X_val_cardinal_c+X_val_decimal_c+X_val_ordinal_c+X_val_time_c
    X_val =X_val_date+X_val_letters+X_val_cardinal+X_val_decimal+X_val_ordinal+X_val_time
    y_val= y_val_date+y_val_letters+y_val_cardinal+y_val_decimal+y_val_ordinal+y_val_time


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

    #load pre-trained word embedding
    from gensim.models import Word2Vec
    emb_model = Word2Vec.load("./word2vecTrainedModel/word2vec2.model")
    
    #extract sentence vector from the pretrained word embedding
    X_ =[]
    y_ = []
    for i in range(len(X)):
        if X[i] == []:
            continue
        try:
            X_.append(emb_model.wv[X[i]])
            y_.append(emb_model.wv[y[i]])
        except:
            pass
        
    #pad vectors for feeding in neural network
    X_vec = []
    y_vec =[]
    for i in range(len(X_)):
        a= X_[i]
        a= np.transpose(a)
        a = pad_sequences(a,padding= 'post',maxlen= maxlen,value= 0.0,dtype= 'float32')
        a= np.transpose(a)
        X_vec.append(a)

        b= y_[i]
        b= np.transpose(b)
        b = pad_sequences(b,padding= 'post',maxlen= maxlen,value= 0.0,dtype= 'float32')
        b= np.transpose(b)
        y_vec.append(b)
        
        
    X_vec = np.array(X_vec)
    y_vec = np.array(y_vec)


    return X_vec,y_vec  #returns padded sample vectors 

            


class textNoiseCGAN():
    
    
    def __init__(self):
        #declare necessary variables
        self.n_units = 256
        self.timestep = 7
        self.embedding_dim = 50    
        self.emb_model= Word2Vec.load("./word2vecTrainedModel/word2vec2.model")


        # Build and compile the discriminator
        D_optimizer = SGD(0.01)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer= D_optimizer,
                                   metrics=['accuracy'])
        #print(self.discriminator.summary())

        # Build the generators
        # Generator takes random noise and sample text vectors as input

        self.Bi_en     = layers.Bidirectional(layers.LSTM(units=self.n_units))
        self.Rep_vect  = layers.RepeatVector(self.timestep)
        self.Bi_de     = layers.Bidirectional(layers.LSTM(units=self.n_units, return_sequences=True))
        self.Final_op  = layers.Dense(self.embedding_dim)

        self.generator = self.build_generator()
        #print(self.generator.summary())


        GAN_text = layers.Input(shape=(self.timestep,self.embedding_dim,))
        GAN_noise= layers.Input(shape=(self.timestep,self.embedding_dim,))
        output_text= self.generator([GAN_text,GAN_noise])
        self.discriminator.trainable = False
        label= self.discriminator([output_text,GAN_text])
        self.GAN = Model([GAN_text,GAN_noise],label)
        GAN_optimizer = SGD(0.01,nesterov= True)
        self.GAN.compile(loss=['binary_crossentropy'],optimizer=GAN_optimizer,metrics='accuracy')
        #print(self.GAN.summary())

        #define GAN network


    def build_generator(self):

        text =  layers.Input(shape=(self.timestep,self.embedding_dim,))
        noise = layers.Input(shape=(self.timestep,self.embedding_dim))
        merged = layers.multiply([text,noise])
        en_out= self.Bi_en(merged)
        de_in = self.Rep_vect(en_out)
        de_out= self.Bi_de(de_in)
        final_op = self.Final_op(de_out)
        g_model= Model(inputs=[text,noise],outputs= final_op)


        return g_model

    def build_discriminator(self):

        text = layers.Input(shape=(self.timestep,self.embedding_dim,))
        context= layers.Input(shape=(self.timestep,self.embedding_dim,))
        concatenated= layers.concatenate([text,context])


        conv1_op = layers.Conv1D(filters=64, kernel_size=5, activation='relu',padding='same')(concatenated)
        ma_pool1 = layers.MaxPooling1D(pool_size=2)(conv1_op)
        conv2_op = layers.Conv1D(64, 3, activation='relu',padding= 'same')(ma_pool1)
        ma_pool2 = layers.MaxPooling1D(pool_size=2)(conv2_op)
        flatt    = layers.Flatten()(ma_pool2)
        d1_out= layers.Dense(16, activation='relu')(flatt)
        d_out= layers.Dense(1, activation='sigmoid')(d1_out)
        d_model= Model(inputs=[text,context],outputs= d_out)


        return d_model


    def train(self,X_nnrmlzd, X_nrmlzd,epochs,batch_size= 128,sample_interval = 100):

        X_train= X_nnrmlzd
        y_train= X_nrmlzd
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))



        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of normalized text
            half_batch = int(batch_size/2)
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            
            d_loss= [0.0, 0.0]
            
            if epoch > 1000:

                # Generate a half batch of new normalized text
                noise_1 =np.random.normal(0,1,(half_batch,self.timestep,self.embedding_dim))
                gen_nrmlzd = self.generator.predict([X_train[idx],noise_1])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([y_train[idx],X_train[idx]], valid[:half_batch])
                d_loss_fake = self.discriminator.train_on_batch([gen_nrmlzd, X_train[idx]], fake[:half_batch])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            idx = np.random.randint(0, X_train.shape[0], 4*batch_size)
            noise=np.random.normal(0,1,(4*batch_size,self.timestep,self.embedding_dim))
            valid = np.ones((4*batch_size, 1))

            # Train the generator
            g_loss = self.GAN.train_on_batch([X_train[idx],noise], valid)

            # Plot the progress
            if epoch % 50 ==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and epoch != 0:
                #set test size 25
                ix = np.random.randint(0,X_train.shape[0],25)
                X_sample_test,y_sample_test= X_train[ix],y_train[ix]
                self.sample_text(epoch,X_sample_test,y_sample_test)
        return 0

    def sample_text(self,epoch,x,y_true):
        noise=np.random.normal(0,1,(x.shape[0],self.timestep,self.embedding_dim))
        y= self.generator.predict([x,noise])
        print('Performance after epoch no: %d' % epoch)
        print('Generator Input Text samples')
        for sample_sen_vec in x:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded.append(c[0][0])
            print(decoded)
        
        print('Ground truth Text samples')
        for sample_sen_vec in y_true:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:        #if confidence is less than 70 then don't add it.
                    decoded.append(c[0][0])
            print(decoded)
        print('Generator Output Text samples')
        for sample_sen_vec in y:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded.append(c[0][0])
            print(decoded)        

if __name__ == '__main__':
    
    cgan = textNoiseCGAN()
    X_train,y_train= Data_preparator(datasize= 200000, timestep = 7)
    cgan.train(X_train,y_train,epochs=2000000, batch_size=128, sample_interval=2000)
            
            
    
