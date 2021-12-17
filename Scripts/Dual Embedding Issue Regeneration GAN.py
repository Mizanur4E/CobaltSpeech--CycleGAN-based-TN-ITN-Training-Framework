class GAN():
    def __init__(self):
        self.vocab_size1 = 18029
        self.vocab_size2 = 2039
        self.maxlen1 = 4 
        self.maxlen2 = 7
        self.embedding_dim = 100
        self.n_units= 256
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        inp1 = layers.Input(shape=(self.maxlen1,))
        gen_out= self.generator(inp1)
        self.discriminator.trainable= False
        gen_out = layers.Lambda(lambda x: K.cast(K.argmax(x), dtype='float32'))(gen_out)
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
        text_y = layers.Input(shape= (self.maxlen2,))
        embed_out = layers.Embedding(self.vocab_size2,self.embedding_dim,input_length=self.maxlen2)(text_y)
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
