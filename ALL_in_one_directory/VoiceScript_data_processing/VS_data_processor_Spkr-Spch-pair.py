
'''
This code preporcess voiceScript data and generate speaker and speecch pair. 
First it remove header and footer's unnecessary texts and then detects speaker and corresponding speech. 
It also able to generate word level speaker and speech pair
'''


import re
import os
import pandas as pd


def footer_remover(file): #header removed text     
    ending_tag = ['Deposition concluded', 'Witness excused', 'Hearing concluded']
    length_file = len(file)
    footer_start  = 0   #start finding footer from this line number
    extracted_text = []

  
    extracted_text = []
    i=footer_start
    ix = 0
    for line in file:
        i+=1

        if i > footer_start and i< length_file :
            if ix == 0:
                 for tag in ending_tag:
                        if re.search(tag,line):
                            ix=i
                            break
            elif ix !=0 :
                break
        else: break
    extracted_text = file[:ix-1]
    
    return extracted_text

def headerFooter_remover(file):
    


    starting_tags_i = ['THE VIDEOGRAPHER','THE COURT REPORTER', 'THE REPORTER', 'THE COURT']  #find these tags and cut lines before it
    starting_tags_ii = ['P R O C E E D I N G S','DEPOSITION OF \w', 'PROCEEDINGS \W', 'EVIDENTIARY HEARING'] #if i type is not found then find these tag
    header_era  = 220                  #find starting point within these lines
    
    
    
    with open(file, errors= 'ignore') as f:
      lines= f.readlines()

    length_file = len(lines)
    extracted_text = []
    i=0
    ix = 0
    for line in lines:
        i+=1

        if i < header_era :
            if ix == 0:
                 for tag in starting_tags_i:
                        if re.search(tag,line):
                            ix=i
                            break
            elif ix !=0 :
                break
        else: break


    i=0
    if ix == 0:
        for line in lines:
            i+=1

            if i < 180 :
                if ix == 0:
                     for tag in starting_tags_ii:
                            if re.search(tag,line):
                                print(tag)
                                ix=i+2
                                break
                elif ix !=0 :
                    break
            else: break


    header_extracted_text =lines[ix-1:] 
    HF_extracted_text = footer_remover(header_extracted_text)
    
    
    #remove line number and unwanted space
    for line in HF_extracted_text:
        line = line[4:]
        extracted_text.append(line)
    return extracted_text





def speaker_speech_pair_generator(a):
    
    speaker_tag = ['Q.','A.']
    speaker = []
    speech = []

    for line in a:
        line= line.strip()
        if line != '' :
            if (re.match('\D+:',line)):
                line = line.split(':')
                speaker.append(line[0].strip())
                speech.append(line[1].strip())
                continue

            for tag in speaker_tag:
                line=line.strip()


                if (re.match(tag,line)):

                    line = line.split(tag)
                    speaker.append(tag)
                    speech.append(line[-1])
                    break

            else:
                #print(line)
                speaker.append(speaker[-1])
                speech.append(line)


    
    return speaker,speech



def word_level_pair_gen(s,sp):  #s -> speaker, sp-> speech

    spkr= []
    spch= []

    for i in range(len(s)):
        arr =sp[i].split(' ')

        for j in range(len(arr)):
            spch.append(arr[j])
            spkr.append(s[i])
            
    return spkr, spch
            


def generate_csv(input_file,path):
    
    b= headerFooter_remover(input_file)
    s,sp= speaker_speech_pair_generator(b)
    s1,sp1= word_level_pair_gen(s,sp)
    df = pd.DataFrame()
    df['Speaker']= s1
    df['speech']=sp1
    title= str(input_file)
    title= title[11:-6]
    title = 'output-' + title +'.csv'
    
    
    df.to_csv(title)

if __name__ == "__main__":
    
    src_dir = '../formatterData/'
    tar_dir =  'ProcessedDataCsv'
    with os.scandir(src_dir) as entries:
        for entry in entries:
            chk = str(entry)
            if chk == "<DirEntry '.ipynb_checkpoints'>":
                continue
            print(entry)
            generate_csv(entry,tar_dir)
            print('completed')
        print('Done!!!')
