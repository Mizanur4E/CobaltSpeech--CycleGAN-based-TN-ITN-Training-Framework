    ''' 
     produces comparison files that will be used to make final dataset
     metal_tiger -> text_cleaing.toml -> sclite
    '''



#This code apply MetalTiger to produce normalized text
def to_normalized_text():
    import os
    src_dir = 'all_text'
    tar_dir = 'all_speech'
    i=0
    count = 1
    with os.scandir(src_dir) as files:

        for file in files:

            nam= str(file)
            file=nam[11:-2]
            nam='speech-form-'+ nam[21:-2]

            file= 'all_text/'+file
            #print(file)
            #print(nam)
            output_file =  'all_speech/'+nam

            if i < count:

                print('started: ',nam)

                ! ./MetalTiger transform -config metaltiger-master/sample_models/lm_clean/en_US/spokenform.toml -input $file \
                        >$output_file


            print('Done upto', nam, i)
            i +=1
        print('Complete!')
    
#This code apply metal tiger's text_cleaning.toml for text cleaning
def text_cleaning():
    
    import os
    src_dir = 'all_text'
    i=0
    count = -1
    with os.scandir(src_dir) as files:

        for file in files:

            nam= str(file)
            file=nam[11:-2]



            #print(file)
            #print(nam)
            output_file =  'all_text_cleaned/'+file
            file = 'all_text/'+file
            if i > count:

                print('started: ',file)

                ! ./MetalTiger transform -config metaltiger-master/sample_models/lm_clean/en_US/text_cleaning.toml -input $file \
                       >$output_file


            print('Done upto', file, i)
            i +=1
        print('Complete!')

    
#Applying sclite to produce comparsion
def make_comparison_files():
    import os
    src_dir = 'all_text_cleaned'
    i=0
    count = -1
    with os.scandir(src_dir) as files:

        for file in files:

            nam= str(file)
            file=nam[21:-2]

            ref =  'all_text_cleaned/text-form-'+file
            hyp = 'all_speech/speech-form-'+file
            tar = 'Comparison-files/comp-'+file
            if i > count:


                !cat $ref | tr '\n' ' ' > single-test-out/ref.txt
                !echo "(gar)" >> single-test-out/ref.txt
                !cat $hyp | tr '\n' ' ' > single-test-out/hyp.txt
                !echo "(gar)" >> single-test-out/hyp.txt

                !./sclite -i swb -r single-test-out/ref.txt -h single-test-out/hyp.txt -o pralign sum -l 140 -O single-test-out
                !cp single-test-out/hyp.txt.pra $tar




            print('Done upto', file, i)
            i +=1
        print('Complete!')

        
if __name__ == "__main__":
    

    
    to_normalized_text()
    text_cleaning()
    make_comparison_files()
