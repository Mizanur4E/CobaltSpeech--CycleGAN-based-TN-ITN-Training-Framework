class vecCycleGAN():
    
    def __init__(self, X,y):
        
        self.training_epochs= 20
        self.batch_size = 128
        self.embedding_dim = 200
        self.maxlen = 7
        self.dim=512
        self.pretrain_discriminator_steps = 7
        self.X = X
        self.y = y
        
        self.discriminator_iterations = 5
        
        '''
        create Gxy, Gyx, Dy, Dx and defines losses, then creates graphs
        for connecting networks
        '''
        self.emb_model= Word2Vec.load("word2vec3.model")
        self.D_optimizer = Adam(1e-4,0.5,0.9)
        self.G_optimizer = RMSprop(1e-4)
      
    
              
        self.Dy = self.build_discriminator()
        self.Dx = self.build_discriminator()
        
        self.Gxy = self.build_generator()
        self.Gyx = self.build_generator()
        
        pass
    
    def build_generator(self):
        
        inputs = layers.Input(shape=(self.maxlen,self.embedding_dim))
        
        inp = layers.Bidirectional(layers.LSTM(units=256))(inputs)
        inp = layers.RepeatVector(self.maxlen)(inp)
        inp =layers.Bidirectional(layers.LSTM(units=256,return_sequences= True))(inp)
        out = layers.TimeDistributed(layers.Dense(self.embedding_dim))(inp)
        
        model = Model(inputs= inputs,outputs= out)
        return model
    
    def ResBlock(self,inputs):
        
        output = layers.LeakyReLU()(inputs)
        output = layers.Conv1D(filters =64,kernel_size=3,padding='same',activation=tf.keras.layers.LeakyReLU())(output)
        output = layers.Conv1D(filters =64,kernel_size=3,padding='same',activation=tf.keras.layers.LeakyReLU())(output)
        return inputs+(0.3*output)
        
    
    def build_discriminator(self):
        
        inputs = layers.Input(shape= (self.maxlen,self.embedding_dim))
        output = layers.Conv1D(filters =64,kernel_size=3,padding='same',activation=tf.keras.layers.LeakyReLU())(inputs)
        output = self.ResBlock(output)
        output = self.ResBlock(output)
        output = self.ResBlock(output)
        output = self.ResBlock(output)
        
        output = layers.Flatten()(output)
        output = layers.Dense(1)(output)
        
        model = Model(inputs= inputs,outputs = output)
        return model
        
     
    

    def get_discriminator_loss(self,real_sample_score,false_sample_score,gradient_penalty):

        real_sample_score = tf.reduce_mean(real_sample_score)
        false_sample_score = tf.reduce_mean(false_sample_score)
        discriminator_loss = -(real_sample_score - false_sample_score) + 10.0*gradient_penalty

        return discriminator_loss


    def get_L2_loss(self, targets,outputs):
        assert targets.shape==outputs.shape

        shape = list(targets.shape)
        loss = tf.reduce_sum(tf.pow(targets - outputs, 2)) / (reduce(lambda x, y: x*y, shape))

        return loss


    def get_gradient_penalty(self, generator_outputs_embedded,real_sample_embedded,dis_fn):

        #set gradient penalty
        alpha = tf.random.uniform(
            shape=[self.batch_size,1,1], 
            minval=0.,
            maxval=1.)

        differences = generator_outputs_embedded - real_sample_embedded
        interpolates = real_sample_embedded + (alpha*differences)
        
        
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            loss_fn = dis_fn(interpolates)
        
        gradients = tape.gradient(loss_fn,interpolates)
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        
        return gradient_penalty

    def pretrain(self):
        
        #use auto-encoder for pretraining generators
        
        self.Gxy.compile(loss='mse',optimizer='adam',metrics='mae')
        self.Gyx.compile(loss='mse',optimizer='adam',metrics='mae')
        
        print('Pretraining Generator: Gxy')
        self.Gxy.fit(X,X,epochs = 15, batch_size = 128,verbose = 1,validation_split= 0.002)
        print('Pretraining Generator: Gyx')
        self.Gyx.fit(y,y, epochs = 15, batch_size = 128,verbose = 1,validation_split= 0.002)
        
        
    
    def train(self):
        
        summary_step = 100
        batch_num = int(self.X.shape[0]/self.batch_size)
        for epoch in range(self.training_epochs):
            
            for batch in range(batch_num):
                
                #train discriminators
                for i in range(self.discriminator_iterations):
                    
                    ix = np.random.randint(0,self.X.shape[0],self.batch_size)
                    X_chunk = self.X[ix]
                    y_false = self.Gxy(X_chunk)
                    y_chunk = self.y[ix]
                    X_false = self.Gyx(y_chunk)

                    with tf.GradientTape() as tape1:

                        d_out_real = self.Dy(y_chunk)
                        d_out_false = self.Dy(y_false)
                        gp = self.get_gradient_penalty(y_false,y_chunk,self.Dy)
                        Dy_loss = self.get_discriminator_loss(d_out_real,d_out_false,gp)
                        
                    grads1 = tape1.gradient(Dy_loss, self.Dy.trainable_weights)    
                    self.D_optimizer.apply_gradients(zip(grads1, self.Dy.trainable_weights))    
                    
                    with tf.GradientTape() as tape2:

                        d_out_real = self.Dx(y_chunk)
                        d_out_false = self.Dx(y_false)
                        gp = self.get_gradient_penalty(X_false,X_chunk,self.Dx)                    
                        Dx_loss = self.get_discriminator_loss(d_out_real,d_out_false,gp)
                    
                    grads2 = tape2.gradient(Dx_loss, self.Dx.trainable_weights)    
                    self.D_optimizer.apply_gradients(zip(grads2, self.Dx.trainable_weights))    
                    
                
                if batch >= self.pretrain_discriminator_steps:
                    
                    #train generator only
                    ix = np.random.randint(0,self.X.shape[0],self.batch_size)
                    X_chunk = self.X[ix]
                    y_chunck = self.y[ix]
                    

                    with tf.GradientTape(persistent= True) as tape3:
                        
                        x2y_outputs = self.Gxy(X_chunk)
                        y2x_outputs = self.Gyx(y_chunk)

                        y2x_rec = self.Gyx(x2y_outputs)
                        x2y_rec = self.Gxy(y2x_outputs)

                        y2x_recon_loss = self.get_L2_loss(X_chunk,y2x_rec)
                        x2y_recon_loss = self.get_L2_loss(y_chunk,x2y_rec)

                        false_y_sample_score = self.Dy(x2y_outputs)
                        false_x_sample_score = self.Dx(y2x_outputs)
                        #print(x2y_outputs.shape,x2y_recon_loss.shape,x2y_rec.shape,false_y_sample_score.shape)
                        #generator x2y                  
                        gen_x2y_loss = 2*(y2x_recon_loss + x2y_recon_loss)-false_y_sample_score
                        gen_y2x_loss = 2*(y2x_recon_loss + x2y_recon_loss)-false_x_sample_score
                        
                    grads3 = tape3.gradient(gen_x2y_loss,self.Gxy.trainable_weights)    
                    self.G_optimizer.apply_gradients(zip(grads3, self.Gxy.trainable_weights))        
                        
                    grads4 = tape3.gradient(gen_y2x_loss,self.Gyx.trainable_weights)    
                    self.G_optimizer.apply_gradients(zip(grads4, self.Gyx.trainable_weights))                     
                
                    #print(gen_x2y_loss.shape,Dx_loss.shape)
                if batch%5 ==0 and batch > self.pretrain_discriminator_steps:
                    
                    Gxy_loss = tf.reduce_sum(gen_x2y_loss)
                    Gyx_loss = tf.reduce_sum(gen_y2x_loss)
                    Dx_loss = tf.reduce_sum(Dx_loss)
                    Dy_loss = tf.reduce_sum(Dy_loss)
                    print ("%d [Dx loss: %.3f, Dy loss: %.3f] [Gxy loss: %.3f, Gyx loss: %.3f]" % (epoch, Dx_loss, Dy_loss, Gxy_loss,Gyx_loss))

                
                if batch%summary_step == 0 and batch > 0:
                    #show some text outputs

                    ix = np.random.randint(0,self.X.shape[0],10)
                    X_sam,y_sam= self.X[ix],self.y[ix]
                    self.sample_text(epoch,batch,X_sam,y_sam)
                    pass
                

    def sample_text(self,epoch,batch,x,y_true):
        
        
        y= self.Gxy.predict(x)
        print('Performance after epoch no: %d,batch no: %d' % (epoch,batch))
        
        
        decoded_x = []
        decoded_y_true=[]
        decoded_y_pred= []
        for sample_sen_vec in x:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded.append(c[0][0])
            decoded_x.append(decoded)
        
        for sample_sen_vec in y_true:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:        #if confidence is less than 70 then don't add it.
                    decoded.append(c[0][0])
            decoded_y_true.append(decoded)
        
        for sample_sen_vec in y:
            decoded= []
            for vec in sample_sen_vec:
                c= self.emb_model.wv.similar_by_vector(vec,topn=1)
                if c[0][1] > 0.7:
                    decoded.append(c[0][0])
            decoded_y_pred.append(decoded)       
            
            
        for i in range(len(decoded_x)):
            print(decoded_x[i],'--',decoded_y_true[i],'--',decoded_y_pred[i])
