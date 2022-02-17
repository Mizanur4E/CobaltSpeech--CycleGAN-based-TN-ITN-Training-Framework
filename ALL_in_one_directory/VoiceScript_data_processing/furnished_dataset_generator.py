'''
generatos text-speech form data pair which is able to feed directly to end to end 
seq2seq model
'''


'''
Algorithm: Declare two list one for storing text form and other for speech form. 
we will go forward by comparing two word lines if we see match then we add them in both list. 
If we find mismatch that means it's the start of the normalization and will store the words in different variable untill we get match words again. 
Once we get matched word we are end of the normalized words. 
We will now add the words which was unmatched as a single sample to text form and speech form.
'''


def LinesExtractor(file):
    
    #step-1 make a single string of speech,text and eval from a whole text file
    import re
    with open(file,'r') as f:
        lines=f.readlines()

    linecount =0

    for line in lines:
        linecount +=1

        
    SpeechLine = ''
    TextLine = ''
    EvalLine = ''

    i=16          #avoid first 16 lines

    while True:

        if (i+3) < linecount: 

            TextLine += lines[i][8:]
            SpeechLine += lines[i+1][8:]
            EvalLine += lines[i+2][8:]

            i +=4
        else: break   
    SpeechLine = re.sub('\n','', SpeechLine)
    TextLine = re.sub('\n','',TextLine)
    EvalLine = re.sub('\n','',EvalLine)
    
    
    return TextLine, SpeechLine, EvalLine

def finalizer(t,s):   #prepares text and speech coloumn from text and speech line

    speech_form = []
    text_form = []

    text_words=  t.split(' ')
    temp = []
    for word in text_words:
        if word != '':
            temp.append(word)

    text_words= temp 

    speech_words= s.split(' ')
    temp = []
    for word in speech_words:
        if word != '':
            temp.append(word)
    speech_words= temp

    is_match = 1


    text_sample= ''
    speech_sample= ''

    for i in range(len(text_words)):

        if text_words[i] == speech_words[i]:

            if is_match == 0:

                speech_form.append(speech_sample)
                text_form.append(text_sample)

            is_match = 1
            text_sample =''   #reinitialize text_sample and speech_sample after one update
            speech_sample= ''
            speech_form.append(text_words[i])
            text_form.append(text_words[i])

        else:
            is_match = 0 
            speech_sample  += ' '+speech_words[i]
            if re.match('\*',text_words[i]):
                continue
            text_sample    += ' '+ text_words[i]



 

    return text_form,speech_form

if __name__ == "__main__":

    import os
    import pandas as pd
    df = pd.DataFrame()

    parenText = []
    parenSpeech = []
    dir= '../formatter/Comparison-files'

    with os.scandir(dir) as entries:
        for entry in entries:

            avoid = str(entry)[11:-2]

            if avoid == '.ipynb_checkpoints':
                continue

            t,s,e = LinesExtractor(entry)
            text, speech= finalizer(t,s)
            if len(text) != len(speech):
                print(len(text),len(speech))
                print(avoid)
                continue

            parenText += text
            parenSpeech += speech

    for i in range(len(parenText)):
        parenText[i] = parenText[i].lower()
        parenSpeech[i]= parenSpeech[i].lower()

    df['Text-form']= parenText
    df['Speech-form']= parenSpeech  
    df.to_csv('FinalDatasetWithContext.csv',index= False)


