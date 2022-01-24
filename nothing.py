class normal():
    
    
    def __init__(self):


        self.timestep=7
        self.embedding_dim=100
        self.n_units = 256
        self.opt = Adam(lr= 1e-4)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.val_acc_fn =  tf.keras.metrics.Accuracy()
        self.train_acc_fn =  tf.keras.metrics.Accuracy()


        inp1= layers.Input(shape=(self.timestep))
        inp= layers.Embedding(input_dim =vocab_size1,output_dim= self.n_units, mask_zero=True)(inp1)
        inp = layers.LSTM(units=256)(inp)
        inp = layers.RepeatVector(self.timestep)(inp)
        inp =layers.LSTM(units=256,return_sequences= True)(inp)
        out = layers.TimeDistributed(layers.Dense(vocab_size2,activation='softmax'))(inp)
        self.model = Model(inputs= inp1,outputs=out)

    def train(self, X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.02) 

        batch_size = 128;
        epochs = 5;
        batch_num = int(X_train.shape[0]/batch_size);

        for i in range(epochs):

            for j in range(batch_num):

                ix = np.random.randint(0,X_train.shape[0],batch_size)
                x_true = X_train[ix]
                y_true = y_train[ix]

                with tf.GradientTape() as tape:
                    y_pred = self.model(x_true,training= True)
                    loss = self.loss_fn(y_true,y_pred)
                grad = tape.gradient(loss, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grad, self.model.trainable_weights)) 
                
                
                
                self.train_acc_fn.update_state(np.argmax(y_true,axis=-1),np.argmax(y_pred,axis =-1))
                train_acc = self.train_acc_fn.result().numpy()
                self.train_acc_fn.reset_states()

                if j% 50 ==0 and j>0:

                    y_pred_test = self.model(X_test)
                    val_loss = self.loss_fn(y_test,y_pred_test)
                    self.val_acc_fn.update_state(np.argmax(y_test,axis=-1),np.argmax(y_pred_test,axis=-1))
                    
                    val_acc = self.val_acc_fn.result().numpy()
                    self.val_acc_fn.reset_states()
                    print('%d, [Training Loss: %.3f, Training_acc: %.2f%%, Val. Loss: %.3f, val. acc: %.2f%%]' % (i+1,loss,train_acc*100,val_loss,val_acc*100))

                    #print("%d,%d, [Training loss: %.3f, Val. loss: %.3f, val. accurcy: %.2f %% ]" % ((i+1),(j+1),loss,val_loss,(val_acc*100)))
