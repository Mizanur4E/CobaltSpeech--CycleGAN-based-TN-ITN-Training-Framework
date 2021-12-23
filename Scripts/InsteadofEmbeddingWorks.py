
"""##Data importing and preparations"""

def Data_preparator1(datasize, timestep):
    
    classwise_data = pd.read_csv(r'./classwiseoutputinputv3.csv')
    """## Preparing train & val set"""
    #print(classwise_data)
    from sklearn.model_selection import train_test_split
    (X_date_c)=classwise_data['X_date_c'].tolist()
    #(X_letters_c)=classwise_data['X_letters_c'] [:152795].tolist()
    (X_cardinal_c)=classwise_data['X_cardinal_c'] [:133744].tolist()
    #(X_verbatim_c)= classwise_data['X_verbatim_c'][:78108].tolist()
    (X_decimal_c)= classwise_data['X_decimal_c'][:9821].tolist()
    (X_measure_c)=classwise_data['X_measure_c'][:14783].tolist()
    #(X_money_c)=classwise_data['X_money_c'][:6128].tolist()
    (X_ordinal_c)=classwise_data['X_ordinal_c'] [:12703].tolist()
    (X_time_c)=classwise_data['X_time_c'] [:1465].tolist()



    (X_date)=classwise_data['X_date'].tolist()
    #(X_letters)=classwise_data['X_letters'] [:152795].tolist()
    (X_cardinal)=classwise_data['X_cardinal'] [:133744].tolist()
    #(X_verbatim)= classwise_data['X_verbatim'][:78108].tolist()
    (X_decimal)= classwise_data['X_decimal'][:9821].tolist()
    (X_measure)=classwise_data['X_measure'][:14783].tolist()
    #(X_money)=classwise_data['X_money'][:6128].tolist()
    (X_ordinal)=classwise_data['X_ordinal'] [:12703].tolist()
    (X_time)=classwise_data['X_time'] [:1465].tolist()



    (y_date)=classwise_data['y_date'].tolist() 
    #(y_letters)=classwise_data['y_letters'] [:152795].tolist()
    (y_cardinal)=classwise_data['y_cardinal']  [:133744].tolist()
    #(y_verbatim)=classwise_data['y_verbatim'][:78108].tolist()
    (y_decimal)=classwise_data['y_decimal'][:9821].tolist()
    (y_measure)=classwise_data['y_measure'][:14783].tolist()
    #(y_money)=classwise_data['y_money'] [:6128].tolist()
    (y_ordinal)=classwise_data['y_ordinal']  [:12703].tolist()
    (y_time)=classwise_data['y_time'] [:1465].tolist()



    # In[3]:




    X_train_date_c, X_val_date_c, X_train_date, X_val_date, y_train_date, y_val_date = train_test_split(X_date_c,X_date, y_date, test_size=0.015, random_state=42)
    #X_train_letters_c, X_val_letters_c, X_train_letters, X_val_letters, y_train_letters, y_val_letters = train_test_split(X_letters_c,X_letters, y_letters, test_size=0.015, random_state=42)
    X_train_cardinal_c, X_val_cardinal_c,X_train_cardinal, X_val_cardinal, y_train_cardinal, y_val_cardinal = train_test_split(X_cardinal_c,X_cardinal, y_cardinal, test_size=0.015, random_state=42)
    #X_train_verbatim_c, X_val_verbatim_c,X_train_verbatim, X_val_verbatim, y_train_verbatim, y_val_verbatim = train_test_split(X_verbatim_c,X_verbatim, y_verbatim, test_size=0.015, random_state=42)
    X_train_decimal_c, X_val_decimal_c,X_train_decimal, X_val_decimal, y_train_decimal, y_val_decimal = train_test_split(X_decimal_c,X_decimal, y_decimal, test_size=0.015, random_state=42)
    X_train_measure_c, X_val_measure_c,X_train_measure, X_val_measure, y_train_measure, y_val_measure = train_test_split(X_measure_c,X_measure, y_measure, test_size=0.015, random_state=42)
    #X_train_money_c, X_val_money_c,X_train_money, X_val_money, y_train_money, y_val_money = train_test_split(X_money_c,X_money, y_money, test_size=0.015, random_state=42)
    X_train_ordinal_c, X_val_ordinal_c,X_train_ordinal, X_val_ordinal, y_train_ordinal, y_val_ordinal = train_test_split(X_ordinal_c, X_ordinal, y_ordinal, test_size=0.015, random_state=42)
    X_train_time_c, X_val_time_c,X_train_time, X_val_time, y_train_time, y_val_time = train_test_split(X_time_c, X_time, y_time, test_size=0.015, random_state=42)
   

    X_train_c =X_train_date_c+X_train_measure_c+X_train_cardinal_c+X_train_decimal_c+X_train_ordinal_c+X_train_time_c
    X_train   =X_train_date+X_train_measure+X_train_cardinal+X_train_decimal+X_train_ordinal+X_train_time
    y_train=   y_train_date+y_train_measure+y_train_cardinal+y_train_decimal+y_train_ordinal+y_train_time

    X_val_c=X_val_date_c+X_val_measure_c+X_val_cardinal_c+X_val_decimal_c+X_val_ordinal_c+X_val_time_c
    X_val =X_val_date+X_val_measure+X_val_cardinal+X_val_decimal+X_val_ordinal+X_val_time
    y_val= y_val_date+y_val_measure+y_val_cardinal+y_val_decimal+y_val_ordinal+y_val_time


    # In[3]:



    print(X_val[:10])
    for i in range(len(X_train)):
        X_train[i] = X_train[i].strip()
        y_train[i] = y_train[i].strip()
        X_train_c[i]= X_train_c[i].strip()



    for i in range(len(X_val)):
        X_val[i] = X_val[i].strip()
        y_val[i] = y_val[i].strip()
        X_val_c[i]= X_val_c[i].strip()
    print(X_val[:10])


    # In[4]:




    X_train_c,X_test_c,X_train,X_test,y_train,y_test = train_test_split( X_train_c,X_train, y_train,test_size=0.0015)

    print(len(X_train_c))
    datasetLength = datasize #ma data sie 673000

    X_nnrmlzd_c= X_train_c[0:datasetLength]
    X_nnrmlzd= X_train[0:datasetLength]
    X_nrmlzd = y_train[0:datasetLength]



    X_train_c,_,X_train,_,y_train, _ = train_test_split( X_nnrmlzd_c,X_nnrmlzd, X_nrmlzd,test_size=0.0000000001) #shuffles all data

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_nnrmlzd =X_train
    X_nrmlzd =y_train


    
    
    maxlen= timestep
    
    #split each samples into words
    X= []
    y= []
    
    for i in range(len(X_nnrmlzd)):
        
        X.append(X_nnrmlzd[i].split())
        y.append(X_nrmlzd[i].split())
    
    tokenizer1 = Tokenizer()
    tokenizer1.fit_on_texts(X)
    
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(y)
    X_seq = tokenizer1.texts_to_sequences(X)
    y_seq = tokenizer2.texts_to_sequences(y)
    
    vocab_size1 = len(tokenizer1.word_index)+1
    vocab_size2 = len(tokenizer2.word_index)+1
    
    maxlen = timestep
    X_seq = pad_sequences(X_seq,maxlen = maxlen)
    y_seq = pad_sequences(y_seq,maxlen= maxlen)
    
    X_enc = to_categorical(X_seq,num_classes= vocab_size1)
    y_enc = to_categorical(y_seq,num_classes = vocab_size2)
    
    return X_enc,y_enc,tokenizer1,tokenizer2,vocab_size1,vocab_size2
    
X_enc,y_enc,tokenizer1,tokenizer2,vocab_size1,vocab_size2= Data_preparator1(200000, 7)

timestep=7
embedding_dim=100
inp1= layers.Input(shape=(timestep,vocab_size1,))
inp= layers.Dense(embedding_dim,use_bias=False)(inp1)
inp = layers.Bidirectional(layers.LSTM(units=256))(inp)
inp = layers.RepeatVector(timestep)(inp)
inp =layers.Bidirectional(layers.LSTM(units=256,return_sequences= True))(inp)
out = layers.TimeDistributed(layers.Dense(vocab_size2,activation='softmax'))(inp)
model = Model(inputs= inp1,outputs=out)
model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics='accuracy')
model.summary()


model.fit(X_enc,y_enc,verbose=1,epochs=100,batch_size=128,validation_split=0.1)
