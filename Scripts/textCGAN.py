class textCGAN():
    
    
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
        output_text= self.generator(GAN_text)
        self.discriminator.trainable = False
        label= self.discriminator([output_text,GAN_text])
        self.GAN = Model(GAN_text,label)
        GAN_optimizer = SGD(0.01,nesterov= True)
        self.GAN.compile(loss=['binary_crossentropy'],optimizer=GAN_optimizer)
        #print(self.GAN.summary())

        #define GAN network


    def build_generator(self):

        text = layers.Input(shape=(self.timestep,self.embedding_dim,))
        en_out= self.Bi_en(text)
        de_in = self.Rep_vect(en_out)
        de_out= self.Bi_de(de_in)
        final_op = self.Final_op(de_out)
        g_model= Model(inputs=text,outputs= final_op)


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
            valid = np.ones((4*batch_size, 1))

            # Train the generator
            g_loss = self.GAN.train_on_batch(X_train[idx], valid)

            # Plot the progress
            if epoch % 100 ==0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and epoch != 0:
                #set test size 25
                ix = np.random.randint(0,X_train.shape[0],25)
                X_sample_test,y_sample_test= X_train[ix],y_train[ix]
                self.sample_text(epoch,X_sample_test,y_sample_test)
        return 0

    def sample_text(self,epoch,x,y_true):
        
        y= self.generator.predict(x)
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
                if c[0][1] > 0.7:
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


            
            
    
