import numpy as np
import tensorflow as tf
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
from keras import backend as K

from tensorflow.keras.layers import Embedding

class DifferntiableEmbedding(Embedding):

    def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_value=-1.0,
               input_length=None,
               **kwargs):
        
        self.mask_value = tf.constant(mask_value)

        super(DifferntiableEmbedding, self).__init__(
            input_dim, output_dim, 
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=False,
            input_length=input_length,
            **kwargs)

    def compute_mask(self, inputs, mask=None):
        if self.mask_value is None:
          return None
        return tf.not_equal(inputs, self.mask_value)
    
    @tf.function
    def call(self, inputs):
        # input.shape = (batch, timestep, vocab)
        # self.embeddings.shape = (vocab, embeddingDim)
        # out.shape = (batch, timestep, embeddingDim)
        out = tf.matmul(inputs, self.embeddings)
        return out
      
      
  class GAN():
    def __init__(self):
        self.vocab_size1 = 17950
        self.vocab_size2 = 2062
        self.maxlen1 = 4 
        self.maxlen2 = 7
        self.embedding_dim = 100
        self.n_units= 256
        self.epochs =10
        self.batch_size=64
        self.datasize = 20000
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
                
        inp1 = layers.Input(shape=(self.maxlen1,))
        gen_out= self.generator(inp1)
        self.discriminator.trainable= False
        dis_out = self.discriminator(gen_out)
        self.gan = Model(inputs= inp1,outputs= dis_out)
        self.gan.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')


        
    def build_generator(self):
        text_x = layers.Input(shape=(self.maxlen1,))
        embed_out = layers.Embedding(self.vocab_size1,self.embedding_dim,input_length=self.maxlen1)(text_x)
        enc_out = layers.Bidirectional(layers.LSTM(256))(embed_out)
        enc_out = layers.RepeatVector(self.maxlen2)(enc_out)
        dec_out = layers.Bidirectional(layers.LSTM(256, return_sequences= True))(enc_out)
        out = layers.TimeDistributed(layers.Dense(self.vocab_size2,activation='softmax'))(dec_out)
        model = Model(inputs= text_x,outputs= out)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
        return model

    def build_discriminator(self):
        text_y = layers.Input(shape= (self.maxlen2,self.vocab_size2,))
        embed_out = DifferntiableEmbedding(self.vocab_size2,self.embedding_dim,input_length=self.maxlen2)(text_y)
        conv_out =layers.Conv1D(32,3,strides=1,padding='same')(embed_out)
        conv_out = layers.Conv1D(32,3,strides=1,padding='same')(conv_out)
        drop = layers.Dropout(0.1)(conv_out)
        conv_out = layers.Conv1D(32,3,strides=1,padding='same')(drop)
        conv_out = layers.Conv1D(32,3,strides=1,padding='same')(conv_out)
        drop = layers.Dropout(0.1)(conv_out)        
        flat = layers.Flatten()(drop)
        out = layers.Dense(1,activation='softmax')(flat)
        model = Model(inputs= text_y,outputs=out)
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
        
        return model    
    
    def train(self,x_train,y_train):
        num_batch = int(self.datasize/self.batch_size)
        valid = np.ones((self.batch_size,1))
        fake = np.zeros((self.batch_size,1))
        for i in range(self.epochs):
            for j in range(num_batch):
                idx= np.random.randint(0,y_train.shape[0],self.batch_size)
                y_real = to_categorical(y_train[idx],num_classes= self.vocab_size2)
                y_fake = self.generator.predict(X_train[idx])
                d_loss1,_ = self.discriminator.train_on_batch(y_real,valid)
                d_loss2,_ = self.discriminator.train_on_batch(y_fake,fake)
                g_loss,_= self.gan.train_on_batch(X_train[idx],valid)
                g_loss1,_= self.generator.test_on_batch(X_train[idx],y_real)
                
                print(d_loss1,d_loss2,g_loss,g_loss1)
                
            
