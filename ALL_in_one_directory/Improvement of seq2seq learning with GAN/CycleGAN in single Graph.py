
class textCycleCGAN():

    
    
    def __init__(self):
        
        #declare necessary variables
        self.n_units = 256
        self.timestep = 7
        self.embedding_dim = 50    
        self.emb_model= Word2Vec.load("./word2vecTrainedModel/word2vec2.model")
        self.lambda_cycle= 10
        self.lambda_id =0.1 * self.lambda_cycle
        self.epochs=10
        self.batch_size= 128
        self.datasize = 200000
        self.batch_num = int(self.datasize/self.batch_size)
        
        
        
        
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminators

        self.Dx = self.build_discriminator()
        self.Dy = self.build_discriminator()
        self.Dx.compile(loss=['binary_crossentropy'],optimizer=optimizer,  metrics=['accuracy'])  
        self.Dy.compile(loss=['binary_crossentropy'],optimizer= optimizer, metrics=['accuracy'])
        
        #Build and compile the generators
        self.Gxy = self.build_generator()
        self.Gyx = self.build_generator()
        
        
        #input texts from both generators
        x= layers.Input(shape=(self.timestep,self.embedding_dim))
        y= layers.Input(shape=(self.timestep,self.embedding_dim))
        
        #transformation using generators
        fake_y = self.Gxy(x)
        fake_x = self.Gyx(y)
        
        #reconstruction using generators
        reconstr_x = self.Gyx(fake_y)
        reconstr_y = self.Gxy(fake_x)
        
        #identity mapping
        x_identity = self.Gyx(x)
        y_identity = self.Gxy(y)
        
        #for combined model only generators will be trained
        self.Dx.trainable = False
        self.Dy.trainable = False
        
        #Discriminators determines validity of translated images
        valid_x = self.Dx(fake_x)
        valid_y = self.Dy(fake_y)
        
        #build combined model
        self.combined = Model(inputs=[x,y],outputs=[valid_x,valid_y,reconstr_x,reconstr_y,x_identity,y_identity])
        self.combined.compile(loss=['binary_crossentropy','binary_crossentropy',
                                   'mae','mae',
                                   'mae','mae'],
                              loss_weights=[1,1,
                                           self.lambda_cycle,self.lambda_cycle,
                                           self.lambda_id,self.lambda_id],optimizer=optimizer)


    def build_generator(self):

        text =  layers.Input(shape=(self.timestep,self.embedding_dim,))
        en_out= layers.Bidirectional(layers.LSTM(units=self.n_units))(text)
        de_in = layers.RepeatVector(self.timestep)(en_out)
        de_out= layers.Bidirectional(layers.LSTM(units=self.n_units, return_sequences=True))(de_in)
        final_op = layers.TimeDistributed(layers.Dense(self.embedding_dim))(de_out)
        g_model= Model(inputs=text,outputs= final_op)


        return g_model

    def build_discriminator(self):

        text = layers.Input(shape=(self.timestep,self.embedding_dim,))
        conv1_op = layers.Conv1D(filters=64, kernel_size=3, activation='relu',padding='same')(text)
        conv1_op = layers.Dropout(0.3)(conv1_op)
        ma_pool1 = layers.MaxPooling1D(pool_size=2)(conv1_op)
        conv2_op = layers.Conv1D(64, 3, activation='relu',padding= 'same')(ma_pool1)
        conv2_op = layers.Dropout(0.3)(conv2_op)
        ma_pool2 = layers.MaxPooling1D(pool_size=2)(conv2_op)
        flatt    = layers.Flatten()(ma_pool2)
        d1_out= layers.Dense(16, activation='relu')(flatt)
        d_out= layers.Dense(1, activation='sigmoid')(d1_out)
        d_model= Model(inputs=text,outputs= d_out)


        return d_model
    
    
    def train(self,x_train,y_train):
        
        valid = np.ones((self.batch_size,1))
        fake = np.zeros((self.batch_size,1))
        
        for i in range(self.epochs):
            
            for j in range(self.batch_num):
                
                idx= np.random.randint(0,y_train.shape[0],self.batch_size)
                y_real = y_train[idx]
                y_fake = self.Gxy.predict(x_train[idx])
                x_real = x_train[idx]
                x_fake = self.Gyx.predict(y_train[idx])
                
                #train discriminators
                dx_loss1= self.Dx.train_on_batch(x_real,valid)
                dx_loss2 = self.Dx.train_on_batch(x_fake,fake)
                dx_loss = 0.5*np.add(dx_loss1,dx_loss2)

                dy_loss1= self.Dy.train_on_batch(y_real,valid)
                dy_loss2=  self.Dy.train_on_batch(y_fake,fake)
                dy_loss = 0.5*np.add(dy_loss1,dy_loss2)
                
                d_loss= 0.5*np.add(dx_loss,dy_loss)
                #train combined model
                g_loss=self.combined.train_on_batch([x_real,y_real],[valid,valid,x_real,y_real,x_real,y_real])
                
                if j % 25 ==0:
                    #print('Epoch[%d],Batch[%d],Dx[%.5f],Dy[%.5f],Combined[%.5f]'%((i+1),(j+1),dx_loss,dy_loss,combined_loss))
                    # Plot the progress
                    print ("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] " \
                                                                        % ( (i+1), 
                                                                            (j+1), 
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6])))
    

                if j % 300 == 0:# and j != 0:
                    #set test size 25
                    ix = np.random.randint(0,x_train.shape[0],15)
                    X_sample_test,y_sample_test= x_train[ix],y_train[ix]
                    self.sample_text(i+1,X_sample_test,y_sample_test)
    
    def sample_text(self,epoch,x,y_true):
        
        
        y_pred= self.Gxy.predict(x)
        print('Performance X->Y after epoch no: %d' % epoch)
        print('Ground truth Text samples- generator output')
        decoded_y_true_full = []
        for sample_sen_vec in y_true:
            decoded_y_true= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:        #if confidence is less than 70 then don't add it.
                    decoded_y_true.append(c[0][0])  
            decoded_y_true_full.append(decoded_y_true)
        decoded_y_pred_full = []    
        for sample_sen_vec in y_pred:
            decoded_y_pred= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded_y_pred.append(c[0][0])
            decoded_y_pred_full.append(decoded_y_pred)          
        for i in range(len(decoded_y_true_full)):
            print(decoded_y_true_full[i],'---',decoded_y_pred_full[i])
           
        
        x_pred= self.Gyx.predict(y_true)
        print('***************************************')
        print('***************************************')
        
        print('Performance Y->X after epoch no: %d' % epoch)
        print('Ground truth Text samples- generator output')
        decoded_x_true_full = []
        for sample_sen_vec in x:
            decoded_x_true= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:        #if confidence is less than 70 then don't add it.
                    decoded_x_true.append(c[0][0])  
            decoded_x_true_full.append(decoded_x_true)
        
        decoded_x_pred_full = []
        for sample_sen_vec in x_pred:
            decoded_x_pred= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded_x_pred.append(c[0][0])
            decoded_x_pred_full.append(decoded_x_pred)  
        for i in range(len(decoded_x_true_full)):
            print(decoded_x_true_full[i],'---',decoded_x_pred_full[i])
           
                        
                
        
