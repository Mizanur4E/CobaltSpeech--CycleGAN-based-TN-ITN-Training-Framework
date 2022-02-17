'''
prepare dataset for IOB tagger 
'''

def IOB_tagger_data_preprocessor():
    
    file = 'Comparison-files/comp-VS-1035-ESQMIA-J7374393.txt'

    #step-1 make a single string of speech,text and eval from a whole text file
    import re
    with open(file,'r') as f:
        lines=f.readlines()

    linecount =0

    for line in lines:
        linecount +=1

    print(linecount)
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

    #step-2 find location of 'I',"S" inside evalLIne

    template = []

    for i in range(len(EvalLine)):
        if EvalLine[i] == 'I' or EvalLine[i] == 'S':
            template.append(i)



    #step-3: start with a 'S' find next 'S' and check if it is the next word of sppechline. if yes then find next else it's the end 



    #textData = []
    #speechData = []

    words= SpeechLine.split(' ')
    tagline= []
    for word in words:
        tagline.append('O')


    i=0

    while True:

        if EvalLine[template[i]] == 'S':

            startpos_text = template[i]
            len_text= 1

            curr_s = i
            b= i+1

            while True:
                if b > (len(template)-1):
                    break
                elif EvalLine[template[b]] == 'S':

                    next_s= b

                    #slice_from_curr_s
                    slices_from_curr = SpeechLine[template[curr_s]:].split(' ')
                    slices_from_next =  SpeechLine[template[next_s]:].split(' ')


                    if slices_from_curr[1]== slices_from_next[0] :
                        curr_s = b
                        len_text +=1
                        b +=1
                        continue
                    else:
                        break
                else:
                    break


            a=i-1
            while True:
                if a < 0:
                    break

                elif EvalLine[template[a]] == 'S':
                    break


                elif EvalLine[template[a]] == 'I':
                    a= a-1
                    continue

            startpos_speech = template[(a+1)]

            len_speech = b-a-1

            #t= findText(TextLine,startpos_text,len_text)
            #s= findText(SpeechLine,startpos_speech, len_speech)

            tagline = findTag(SpeechLine,tagline,startpos_speech,len_speech)
            #textData.append(t)
            #speechData.append(s)    

            i = b

        else:
            i +=1

        if i > (len(template)-1):
            break
            
def findTag(SpeechLine,tagline,startpos_speech,len_speech):
    prev_words= SpeechLine[:startpos_speech].split(' ')
 
    n= len(prev_words)
    tagline[n]='B'
    for i in range(1,len_speech,1):
        tagline[(n+i)] = 'I'        
    
    return tagline
  
if __name__ == "__main__":
  
  IOB_tagger_data_preprocessor()
