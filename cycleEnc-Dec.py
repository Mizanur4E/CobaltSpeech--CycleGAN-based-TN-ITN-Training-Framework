class Cycle():
   
    ''' Implements cycle enc-dec where Gxy and Gyx are trained with cycle loss and x-y seq loss
    Gxy_loss = Gen_loss + x-y_categorical loss where 
    Gen_loss = xyx_reconstruction loss + yxy_reconstruction_loss'''
    def __init__(self):
        
        self.timestep = 7
        self.timestep=7
        self.embedding_dim=100
        self.n_units = 256
        self.opt1 = Adam(lr= 1e-4)
        self.opt2 = Adam(lr= 1e-4)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.val_acc_fn =  tf.keras.metrics.Accuracy()
        self.train_acc_fn =  tf.keras.metrics.Accuracy()

        self.Gxy = self.build_Gxy();
        self.Gyx = self.build_Gyx();
    
    
    def build_Gxy(self):
        
        inp1= layers.Input(shape=(self.timestep))
        inp= layers.Embedding(input_dim =vocab_size1,output_dim= self.n_units, mask_zero=True)(inp1)
        inp = layers.LSTM(units=256)(inp)
        inp = layers.RepeatVector(self.timestep)(inp)
        inp =layers.LSTM(units=256,return_sequences= True)(inp)
        out = layers.TimeDistributed(layers.Dense(vocab_size2,activation='softmax'))(inp)
        model = Model(inputs= inp1,outputs=out)
        return model
    
    
    def build_Gyx(self):
        
        inp1= layers.Input(shape=(self.timestep))
        inp= layers.Embedding(input_dim =vocab_size2,output_dim= self.n_units, mask_zero=True)(inp1)
        inp = layers.LSTM(units=256)(inp)
        inp = layers.RepeatVector(self.timestep)(inp)
        inp =layers.LSTM(units=256,return_sequences= True)(inp)
        out = layers.TimeDistributed(layers.Dense(vocab_size1,activation='softmax'))(inp)
        model = Model(inputs= inp1,outputs=out)
        return model


    def train(self, X,y):

        '''X,y are both of shape (data_size,timestep) 
        '''
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.02) 
        x_test_enc = to_categorical(X_test, num_classes= vocab_size1)
        y_test_enc = to_categorical(y_test,num_classes= vocab_size2)        
        batch_size = 128;
        epochs = 100;
        batch_num = int(X_train.shape[0]/batch_size);

        for i in range(epochs):

            for j in range(batch_num):

                ix = np.random.randint(0,X_train.shape[0],batch_size)
                
                x_true = X_train[ix]
                y_true = y_train[ix]
                
                x_true_enc = to_categorical(x_true, num_classes= vocab_size1)
                y_true_enc = to_categorical(y_true,num_classes= vocab_size2)
                
   

                with tf.GradientTape(persistent= True) as tape:
                    
                    x2y_out = self.Gxy(x_true ,training = True)
                    y2x_out = self.Gyx(y_true,training = True)
                    
                    x2y_out_sh = np.argmax(x2y_out,axis = -1)
                    y2x_out_sh = np.argmax(y2x_out,axis = -1)
                    
                    y2x_rec = self.Gyx(x2y_out_sh, training = True)
                    x2y_rec = self.Gxy(y2x_out_sh, training = True)
                    
                    xyx_recon_loss = self.loss_fn(x_true_enc,y2x_rec)
                    yxy_recon_loss = self.loss_fn(y_true_enc,x2y_rec)
                    gen_loss = xyx_recon_loss + yxy_recon_loss
                    
                    xy_loss = self.loss_fn(y_true_enc,x2y_out)
                    yx_loss = self.loss_fn(x_true_enc,y2x_out)
                    Gxy_loss = gen_loss + xy_loss
                    Gyx_loss = gen_loss + yx_loss
                    
                    

                grad1 = tape.gradient(Gxy_loss, self.Gxy.trainable_weights)
                self.opt1.apply_gradients(zip(grad1, self.Gxy.trainable_weights)) 
                
                grad2 = tape.gradient(Gyx_loss, self.Gyx.trainable_weights)
                self.opt2.apply_gradients(zip(grad2, self.Gyx.trainable_weights)) 
                

                '''self.train_acc_fn.update_state(np.argmax(y_true,axis=-1),np.argmax(y_pred,axis =-1))
                train_acc = self.train_acc_fn.result().numpy()
                self.train_acc_fn.reset_states()'''
                
                

                if j% 50 ==0 and j>0:
                    ix = np.random.randint(0,X_test.shape[0],X_test.shape[0])
                    y_pred_test = self.Gxy(X_test[ix])
                    val_loss_xy = self.loss_fn(y_test_enc[ix],y_pred_test)
                    print(xy_loss,yx_loss)
                    x_pred_test = self.Gyx(y_test[ix])
                    val_loss_yx = self.loss_fn(x_test_enc[ix],x_pred_test)
                    print('%d, [Training Loss: %.3f, Val. Loss_XY: %.3f, Val. Loss_YX: %.3f]' % (i+1,Gxy_loss,val_loss_xy,val_loss_yx))

                    #print("%d,%d, [Training loss: %.3f, Val. loss: %.3f, val. accurcy: %.2f %% ]" % ((i+1),(j+1),loss,val_loss,(val_acc*100)))
        
