class NoEmbeddingGAN():
    
    def __init__(self, vocab_size1,vocab_size2,tokenizer1,tokenizer2):
        
        self.timestep = 7
        self.embedding_dim = 100
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        optimizer = Adam(0.0002,0.9)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        
        inp1 = layers.Input(shape=(self.timestep,self.vocab_size1,))
        gen_out = self.generator(inp1)
        self.discriminator.trainable = False
        dis_out = self.discriminator([gen_out,inp1])
        self.gan = Model(inputs= inp1,outputs= dis_out)
        self.gan.compile(loss='binary_crossentropy',optimizer=optimizer,metrics='accuracy')
        
        
        
    def build_generator(self):
        
        inp1= layers.Input(shape=(self.timestep,self.vocab_size1,))
        inp= layers.Dense(self.embedding_dim,use_bias=False)(inp1)
        inp = layers.Bidirectional(layers.LSTM(units=256))(inp)
        inp = layers.RepeatVector(self.timestep)(inp)
        inp =layers.Bidirectional(layers.LSTM(units=256,return_sequences= True))(inp)
        out = layers.TimeDistributed(layers.Dense(self.vocab_size2,activation='softmax'))(inp)
        model = Model(inputs= inp1,outputs=out)
        model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics='accuracy')
        
        return model
    
    def build_discriminator(self):
        
        inp1 = layers.Input(shape=(self.timestep,self.vocab_size2,)) #inputs seq
        den_inp1 = layers.Dense(self.embedding_dim,use_bias=False)(inp1) #contains embedding
        
        condtn = layers.Input(shape= (self.timestep,self.vocab_size1,))  #condition
        den_condtn= layers.Dense(self.embedding_dim,use_bias=False)(condtn)
        
        merged = layers.concatenate([den_inp1,den_condtn],axis =1)
        conv = layers.Conv1D(64,4,padding='valid')(merged)
        conv = layers.Dropout(0.3)(conv)
        pool = layers.MaxPooling1D(2)(conv)
        flat = layers.Flatten()(pool)
        den = layers.Dense(10)(flat)
        out = layers.Dense(1,activation='sigmoid')(den)
        
        model = Model(inputs=[inp1,condtn],outputs= out)
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
        
        return model
        
    def pretrain(self,X_nnrmlzd, X_nrmlzd,epochs):
        self.generator.fit(X_nnrmlzd,X_nrmlzd,epochs= 2, batch_size = 128,verbose= 1,validation_split=0.1)


    def train(self,X_nnrmlzd, X_nrmlzd,epochs,sample_interval,batch_size= 128):

        X_train= X_nnrmlzd  #embedding_vector
        y_train= X_nrmlzd   #one
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



            # Generate a half batch of new normalized text
            #noise_1 =np.random.normal(0,1,(half_batch,self.timestep,self.embedding_dim))
            gen_nrmlzd = self.generator.predict(X_train[idx])


            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([y_train[idx],X_train[idx]], valid[:half_batch])
            d_loss_fake = self.discriminator.train_on_batch([gen_nrmlzd, X_train[idx]], fake[:half_batch])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            idx = np.random.randint(0, X_train.shape[0], 4*batch_size)
            #noise=np.random.normal(0,1,(4*batch_size,self.timestep,self.embedding_dim))
            valid = np.ones((4*batch_size, 1))

            # Train the generator
            g_loss = self.gan.train_on_batch(X_train[idx], valid)

            # Plot the progress
            if epoch % 50 ==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                #set test size 25
                ix = np.random.randint(0,X_train.shape[0],25)
                X_sample_test,y_sample_test= X_train[ix],y_train[ix]
                
                self.generator.evaluate(X_sample_test,y_sample_test)
                #self.sample_text(epoch,X_sample_test,y_sample_test)
        return 0

    def sample_text(self,epoch,x,y_true):
        
        #noise=np.random.normal(0,1,(x.shape[0],self.timestep,self.embedding_dim))
        y= self.generator.predict(x)
        y = np.argmax(y,axis = -1)
        y_true= np.argmax(y_true,axis=-1)
        y_true_text = self.tokenizer2.sequences_to_texts(y_true)
        y_pred_text = self.tokenizer2.sequences_to_texts(y)
        
        x_text = self.tokenizer1.sequences_to_texts(x)
        
        for i in range(len(x_text)):
            print(x_text[i],'--',y_true_text[i],'--',y_pred_text[i])
        
        
        
        
