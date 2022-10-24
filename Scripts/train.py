        for i in range(epochs):

            for j in range(batch_num):
                
                #train discriminators
                for _  in range(self.discriminator_iter):
                    
                    ix =  np.random.randint(0,X_train.shape[0],batch_size)
                    x_true = X_train[ix]
                    y_true = y_train[ix]                   
                    
                    fake_x = self.Gyx(y_true)
                    fake_y = self.Gxy(x_true)
                    
                    fake_x = np.argmax(fake_x,axis =-1)
                    fake_y = np.argmax(fake_y,axis=-1)
                    
                    Dx_loss_fake = self.Dx.train_on_batch(fake_x,fake)
                    Dx_loss_valid = self.Dx.train_on_batch(x_true,valid)
                    
                    Dy_loss_fake = self.Dy.train_on_batch(fake_y,fake)
                    Dy_loss_valid = self.Dy.train_on_batch(y_true,valid)                
                
                
                #train generators
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
                    
                    
                    fake_x_score = np.sum(self.Dx(y2x_out_sh,training=True))
                    fake_y_score = np.sum(self.Dy(x2y_out_sh,training= True))
                    
                    xy_loss = self.loss_fn(y_true_enc,x2y_out)
                    yx_loss = self.loss_fn(x_true_enc,y2x_out)
                    Gxy_loss = 2*(gen_loss + xy_loss) - fake_y_score
                    Gyx_loss = 2*(gen_loss + yx_loss) - fake_x_score
                    
                    

                grad1 = tape.gradient(Gxy_loss, self.Gxy.trainable_weights)
                self.opt1.apply_gradients(zip(grad1, self.Gxy.trainable_weights)) 
                
                grad2 = tape.gradient(Gyx_loss, self.Gyx.trainable_weights)
                self.opt2.apply_gradients(zip(grad2, self.Gyx.trainable_weights)) 
