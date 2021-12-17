
            


class textCycleCGAN():

    
    
    def __init__(self):
        #declare necessary variables
        self.n_units = 256
        self.timestep = 7
        self.embedding_dim = 50    
        self.emb_model= Word2Vec.load("./word2vecTrainedModel/word2vec2.model")


        # Build and compile the discriminator
        D_optimizer =  SGD(0.0005,momentum= 0.9,nesterov= True)
        self.Dx = self.build_discriminator()
        self.Dy = self.build_discriminator()
        self.Dx.compile(loss=['binary_crossentropy'],optimizer= D_optimizer,  metrics=['accuracy'])  
        self.Dy.compile(loss=['binary_crossentropy'],optimizer= D_optimizer, metrics=['accuracy'])
  

        # Build the generators

        self.Bi_en     = layers.Bidirectional(layers.LSTM(units=self.n_units))
        self.Rep_vect  = layers.RepeatVector(self.timestep)
        self.Bi_de     = layers.Bidirectional(layers.LSTM(units=self.n_units, return_sequences=True))
        self.Final_op  = layers.Dense(self.embedding_dim)

        self.Gxy = self.build_generator()
        self.Gyx = self.build_generator()

        self.XYX_d = self.composite_XYX()
        self.YXY_d = self.composite_YXY()
      
    def composite_XYX(self):

        #make x-y-x cycle and y-x-y cycle graph
        text_x = layers.Input(shape(self.timestep,self.embedding_dim))
        gXY_out = self.Gxy(text_x)
        self.Gyx.trainable = False
        gYX_out = self.Gyx(gXY_out)
        self.Dy.trainable = False
        dis_out_y = self.Dy(gXY_out)
        composite_XYX = Model(inputs=text_x,outputs=[gYX_out,dis_out_y])
        composite_XYX.compile(loss=['mse','binary_crossentropy'],metrics=['accuracy'],optimizer='adam')

        return composite_XYX

    def composite_YXY(self):

        #make y-x-y cycle and y-x-y cycle graph
        text_y = layers.Input(shape(self.timestep,self.embedding_dim))
        gYX_out = self.Gyx(text_y)
        self.Gxy.trainable = False
        gXY_out = self.Gxy(gYX_out)
        self.Dx.trainable = False
        dis_out_x = self.Dx(gYX_out)
        composite_YXY = Model(inputs=text_y,outputs=[gXY_out,dis_out_x])
        composite_YXY.compile(loss=['mse','binary_crossentropy'],metrics=['accuracy'],optimizer='adam')

        return composite_YXY



    def build_generator(self):

        text =  layers.Input(shape=(self.timestep,self.embedding_dim,))
        en_out= self.Bi_en(text)
        de_in = self.Rep_vect(en_out)
        de_out= self.Bi_de(de_in)
        final_op = self.Final_op(de_out)
        g_model= Model(inputs=text,outputs= final_op)


        return g_model

    def build_discriminator(self):

        text = layers.Input(shape=(self.timestep,self.embedding_dim,))
        conv1_op = layers.Conv1D(filters=64, kernel_size=5, activation='relu',padding='same')(text)
        ma_pool1 = layers.MaxPooling1D(pool_size=2)(conv1_op)
        conv2_op = layers.Conv1D(64, 3, activation='relu',padding= 'same')(ma_pool1)
        ma_pool2 = layers.MaxPooling1D(pool_size=2)(conv2_op)
        flatt    = layers.Flatten()(ma_pool2)
        d1_out= layers.Dense(16, activation='relu')(flatt)
        d_out= layers.Dense(1, activation='sigmoid')(d1_out)
        d_model= Model(inputs=text,outputs= d_out)


        return d_model

    #first train discriminator Dx and Dy, then train composite models: composite_XYX() and composite_YXY()
    #Need to complete,issues with selecting loss for generator seq2seq loss,optimizer and hyperparameter
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

    
