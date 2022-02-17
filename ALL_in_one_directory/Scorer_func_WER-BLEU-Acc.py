

import tensorflow as tf
from tensorflow import keras
import numpy 
import pandas as pd
from numpy.random import randint
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import gc
from jiwer import wer 




def average_bleu(refa, hypa):
  smoothie = SmoothingFunction().method2
  n= len(refa)
  net =0
  for i in range(n):
    
    a= sentence_bleu([refa[i]],hypa[i], smoothing_function=smoothie)
    net= net + a
  me = net/n
  return me

def accuracy_printerXY(X,y_true,gen):
  inp= tokenizer1.texts_to_sequences(X)
  inp =pad_sequences(inp, padding='post', maxlen=maxlen1)
  out = gen.predict(inp)
  out = numpy.argmax(out,axis=-1)
  text_out= tokenizer2.sequences_to_texts(out)
  ground_truth =y_true.to_numpy()
  
  ground_truth_seq = tokenizer2.texts_to_sequences(ground_truth)
  ground_truth_seq =pad_sequences(ground_truth_seq, padding='post', maxlen=maxlen2)
  cal = tf.keras.metrics.Accuracy()
  cal.update_state(out,ground_truth_seq)
  text_out= numpy.asarray(text_out)
  ground_truth= ground_truth.tolist()
  text_out= text_out.tolist()

  #print(text_out[:10])
  #print('------------------------------')
  #print(ground_truth[:10])

  w= wer(ground_truth,text_out)
  print('WER is:' ,(1-w))
  print('Accuracy is:',cal.result().numpy())
  smoothie = SmoothingFunction().method4
  score = average_bleu(ground_truth,text_out)
  print('Average BLEU score is: ',score)
  print('\n')


def accuracy_calcYX(X,y_true,gen):
  inp= tokenizer2.texts_to_sequences(X)
  inp =pad_sequences(inp, padding='post', maxlen=maxlen2)
  out = gen.predict(inp)
  out = numpy.argmax(out,axis=-1)
  text_out= tokenizer1.sequences_to_texts(out)
  ground_truth =y_true.to_numpy()
  text_out= numpy.asarray(text_out)
  
  ground_truth_seq = tokenizer1.texts_to_sequences(ground_truth)
  ground_truth_seq = pad_sequences(ground_truth_seq, padding='post', maxlen=maxlen1)
  cal = tf.keras.metrics.Accuracy()
  cal.update_state(out,ground_truth_seq)
  acc= cal.result().numpy()
  ground_truth= ground_truth.tolist()
  text_out= text_out.tolist()
  #print(text_out[:10])
  #print('------------------------------')
  #print(ground_truth[:10])
  w= wer(ground_truth,text_out)
  
  smoothie = SmoothingFunction().method4
  score = average_bleu(ground_truth,text_out)
  
  return w, acc,score

def accuracy_printerYX(X,y,gen):

  s_wer=0
  s_acc=0
  s_bleu=0
  l = len(X)
  if l < 40: 
    wer,acc, bleu = accuracy_calcYX(X,y,gen) 
      
    print('WER is:', (1-wer))
    print('Acc is:', acc)
    print('BLEU score is:',bleu)
    return 
  else: 
    n= int(l/40)+1
  
    for i in range(n):
        if (i == (n-1)):
          
          wer,acc, bleu = accuracy_calcYX(X[i*40:l],y[i*40:l],gen)
          
        else :
          wer,acc, bleu = accuracy_calcYX(X[i*40:(i+1)*40],y[i*40:(i+1)*40],gen)
          #print(wer)

        
          s_wer = s_wer + wer 
          s_acc = s_acc + acc	
          s_bleu = s_bleu + bleu
        
    
    a_wer= s_wer/(l/40)
    a_bleu= s_bleu/(l/40)
    a_acc= s_acc/(l/40)
   

    print('WER is:', (1-a_wer))
    print('BLEU score is:',a_bleu)
    print('Accuracy is:',a_acc)




def scorer():
    
    classwise_data = pd.read_csv(r'classwiseoutputinputv3.csv')

    (X_date_c)=classwise_data['X_date_c']
    (X_letters_c)=classwise_data['X_letters_c'] [:152795]
    (X_cardinal_c)=classwise_data['X_cardinal_c'] [:133744]
    (X_verbatim_c)= classwise_data['X_verbatim_c'][:78108]
    (X_decimal_c)= classwise_data['X_decimal_c'][:9821]
    (X_measure_c)=classwise_data['X_measure_c'][:14783]
    (X_money_c)=classwise_data['X_money_c'][:6128]
    (X_ordinal_c)=classwise_data['X_ordinal_c'] [:12703]
    (X_time_c)=classwise_data['X_time_c'] [:1465]
    (X_electronic_c)=classwise_data['X_electronic_c'] [:5162]
    (X_digit_c)=classwise_data['X_digit_c'] [:5442]
    (X_fraction_c)=classwise_data['X_fraction_c'] [:1196]
    (X_telephone_c)=classwise_data['X_telephone_c'][:4024]
    (X_address_c)=classwise_data['X_address_c'] [:522]


    (X_date)=classwise_data['X_date']
    (X_letters)=classwise_data['X_letters'] [:152795]
    (X_cardinal)=classwise_data['X_cardinal'] [:133744]
    (X_verbatim)= classwise_data['X_verbatim'][:78108]
    (X_decimal)= classwise_data['X_decimal'][:9821]
    (X_measure)=classwise_data['X_measure'][:14783]
    (X_money)=classwise_data['X_money'][:6128]
    (X_ordinal)=classwise_data['X_ordinal'] [:12703]
    (X_time)=classwise_data['X_time'] [:1465]
    (X_electronic)=classwise_data['X_electronic'] [:5162]
    (X_digit)=classwise_data['X_digit'] [:5442]
    (X_fraction)=classwise_data['X_fraction'] [:1196]
    (X_telephone)=classwise_data['X_telephone'][:4024]
    (X_address)=classwise_data['X_address'] [:522]



    (y_date)=classwise_data['y_date'] 
    (y_letters)=classwise_data['y_letters'] [:152795]
    (y_cardinal)=classwise_data['y_cardinal']  [:133744]
    (y_verbatim)=classwise_data['y_verbatim'][:78108]
    (y_decimal)=classwise_data['y_decimal'][:9821]
    (y_measure)=classwise_data['y_measure'][:14783]
    (y_money)=classwise_data['y_money'] [:6128]
    (y_ordinal)=classwise_data['y_ordinal']  [:12703]
    (y_time)=classwise_data['y_time'] [:1465]
    (y_electronic)=classwise_data['y_electronic']  [:5162]
    (y_digit)=classwise_data['y_digit']  [:5442]
    (y_fraction)=classwise_data['y_fraction'][:1196]
    (y_telephone)=classwise_data['y_telephone'] [:4024]
    (y_address)=classwise_data['y_address'] [:522]


    X_train_date_c, X_test_date_c, X_train_date, X_test_date, y_train_date, y_test_date = train_test_split(X_date_c,X_date, y_date, test_size=0.015, random_state=42)
    X_train_letters_c, X_test_letters_c, X_train_letters, X_test_letters, y_train_letters, y_test_letters = train_test_split(X_letters_c,X_letters, y_letters, test_size=0.015, random_state=42)
    X_train_cardinal_c, X_test_cardinal_c,X_train_cardinal, X_test_cardinal, y_train_cardinal, y_test_cardinal = train_test_split(X_cardinal_c,X_cardinal, y_cardinal, test_size=0.015, random_state=42)
    X_train_verbatim_c, X_test_verbatim_c,X_train_verbatim, X_test_verbatim, y_train_verbatim, y_test_verbatim = train_test_split(X_verbatim_c,X_verbatim, y_verbatim, test_size=0.015, random_state=42)
    X_train_decimal_c, X_test_decimal_c,X_train_decimal, X_test_decimal, y_train_decimal, y_test_decimal = train_test_split(X_decimal_c,X_decimal, y_decimal, test_size=0.015, random_state=42)
    X_train_measure_c, X_test_measure_c,X_train_measure, X_test_measure, y_train_measure, y_test_measure = train_test_split(X_measure_c,X_measure, y_measure, test_size=0.015, random_state=42)
    X_train_money_c, X_test_money_c,X_train_money, X_test_money, y_train_money, y_test_money = train_test_split(X_money_c,X_money, y_money, test_size=0.015, random_state=42)
    X_train_ordinal_c, X_test_ordinal_c,X_train_ordinal, X_test_ordinal, y_train_ordinal, y_test_ordinal = train_test_split(X_ordinal_c, X_ordinal, y_ordinal, test_size=0.015, random_state=42)
    X_train_time_c, X_test_time_c,X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time_c, X_time, y_time, test_size=0.015, random_state=42)
    X_train_electronic_c, X_test_electronic_c,X_train_electronic, X_test_electronic, y_train_electronic, y_test_electronic = train_test_split(X_electronic_c, X_electronic, y_electronic, test_size=0.015, random_state=42)
    X_train_digit_c, X_test_digit_c,X_train_digit, X_test_digit, y_train_digit, y_test_digit = train_test_split(X_digit_c,X_digit, y_digit, test_size=0.015, random_state=42)
    X_train_fraction_c, X_test_fraction_c, X_train_fraction, X_test_fraction, y_train_fraction, y_test_fraction = train_test_split(X_fraction_c, X_fraction, y_fraction, test_size=0.015, random_state=42)
    X_train_telephone_c, X_test_telephone_c,X_train_telephone, X_test_telephone, y_train_telephone, y_test_telephone = train_test_split(X_telephone_c,X_telephone, y_telephone, test_size=0.015, random_state=42)
    X_train_address_c, X_test_address_c,X_train_address, X_test_address, y_train_address, y_test_address = train_test_split(X_address_c, X_address, y_address, test_size=0.015, random_state=42)

    with open('tokenizer1V2.txt') as f:
      lines1= f.read()

    with open('tokenizer2V2.txt') as f:
      lines2= f.read()

    tokenizer1 = tokenizer_from_json(lines1)
    tokenizer2 = tokenizer_from_json(lines2)

    maxlen1 = 6
    maxlen2 = 9  
    genXY = tf.keras.models.load_model('modelXY_V2')
    genYX = tf.keras.models.load_model('modelYX_V2')


    print('X->Y performance, Normalizer')
    print('For Class: Date')
    accuracy_printerXY(X_test_date_c,y_test_date,genXY)
    print('For Class: Letters')
    accuracy_printerXY(X_test_letters_c,y_test_letters,genXY)
    print('For Class: Cardinal')
    accuracy_printerXY(X_test_cardinal_c,y_test_cardinal,genXY)
    print('For Class: Verbatim')
    accuracy_printerXY(X_test_verbatim_c,y_test_verbatim,genXY)
    print('For Class: Decimal')
    accuracy_printerXY(X_test_decimal_c,y_test_decimal,genXY)
    print('For Class: Measure')
    accuracy_printerXY(X_test_measure_c,y_test_measure,genXY)
    print('For Class: Money')
    accuracy_printerXY(X_test_money_c,y_test_money,genXY)
    print('For Class: Ordinal')
    accuracy_printerXY(X_test_ordinal_c,y_test_ordinal,genXY)
    print('For Class: Time')
    accuracy_printerXY(X_test_time_c,y_test_time,genXY)
    print('For Class: Electronic')
    accuracy_printerXY(X_test_electronic_c,y_test_electronic,genXY)
    print('For Class: Digit')
    accuracy_printerXY(X_test_digit_c,y_test_digit,genXY)
    print('For Class: Fraction')
    accuracy_printerXY(X_test_fraction_c,y_test_fraction,genXY)
    print('For Class: Telephone')
    accuracy_printerXY(X_test_telephone_c,y_test_telephone,genXY)
    print('For Class: Address')
    accuracy_printerXY(X_test_address_c,y_test_address,genXY)

    gc.collect()

    print('Y->X performance, formatter')


    print('For Class: Date')
    accuracy_printerYX(y_test_date,X_test_date,genYX)
    print('For Class: Letters')
    accuracy_printerYX(y_test_letters,X_test_letters,genYX)
    print('For Class: Cardinal')
    accuracy_printerYX(y_test_cardinal,X_test_cardinal,genYX)
    print('For Class: Verbatim')
    accuracy_printerYX(y_test_verbatim,X_test_verbatim,genYX)
    print('For Class: Decimal')
    accuracy_printerYX(y_test_decimal,X_test_decimal,genYX)
    print('For Class: Measure')
    accuracy_printerYX(y_test_measure,X_test_measure,genYX)
    print('For Class: Money')
    accuracy_printerYX(y_test_money,X_test_money,genYX)
    print('For Class: Ordinal')
    accuracy_printerYX(y_test_ordinal,X_test_ordinal,genYX)
    print('For Class: Time')
    accuracy_printerYX(y_test_time,X_test_time,genYX)
    print('For Class: Electronic')
    accuracy_printerYX(y_test_electronic,X_test_electronic,genYX)
    print('For Class: Digit')
    accuracy_printerYX(y_test_digit,X_test_digit,genYX)
    print('For Class: Fraction')
    accuracy_printerYX(y_test_fraction,X_test_fraction,genYX)
    print('For Class: Telephone')
    accuracy_printerYX(X_test_telephone_c,y_test_telephone,genYX)
    print('For Class: Address')
    accuracy_printerYX(X_test_address_c,y_test_address,genYX)
    
if __name__ == "__main__":
    scorer()
