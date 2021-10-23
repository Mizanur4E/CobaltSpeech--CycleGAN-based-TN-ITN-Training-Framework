
Commands for communication
--------------------------------
ssh nayan@bolt.in.cobaltspeech.com      
jupyter notebook --no-browser --port=1234

ssh -L 8080:localhost:1234 nayan@100.88.210.1

http://localhost:8080/



Readme:
----------------------
first initiate ssh communication and enter the bolt server by typing: 
ssh nayan@bolt.in.cobaltspeech.com 

Then open jupyter notebook in no browser mode at port 1234 by the following command:
jupyter notebook --no-browser --port=1234


Now open a new command window from your local machine and connect with server using the following command:

ssh -L 8080:localhost:1234 nayan@100.88.210.1

where nayan is your server username and 100.88.210.1 is your server ip address. It means from server's 1234 port collect data packet and show it on your 8080 port. Now we'll navigate to our local machine's port 8080 by typing following command in cmd window:
http://localhost:8080/


File Transferring:
----------------------
scp classwiseoutputinputv3.csv nayan@100.88.210.1:/home/nayan
scp espresso_tt.py nayan@100.88.210.1:/home/nayan
scp load_and_predict_words.py nayan@100.88.210.1:/home/nayan

File importing from remote to local:
------------------------------------
scp -r nayan@100.88.210.1:/home/nayan/model

Removing a file:
----------------------
rm -rf model/


Local file management:(VoiceScript Data)
-------------------------------
dataset location: /home/alok/voicescript-data/downloaded-data

#see Extracting VoiceScript text.py for details



For running any script:
-----------------------
nvidia-smi          #checking GPU


Openning a text file in command window:
----------------------------------------------
head -n -0 VS-625-TP-103177.txt

here -0 argument shows all the lines. if you put 100 instead of -0 it will print first 100 lines



opening jupyter notebook:
---------------------------------
LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" PATH="/usr/local/cuda-11.2/bin:$PATH" CUDA_VISIBLE_DEVICES="0" jupyter notebook --no-browser --port=1234

Running python script direct from shell:
-----------------------------------------------
LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" PATH="/usr/local/cuda-11.2/bin:$PATH" CUDA_VISIBLE_DEVICES="0" python3 espresso_tt.py


LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" PATH="/usr/local/cuda-11.2/bin:$PATH" CUDA_VISIBLE_DEVICES="1" python3 load_and_predict_words.py



MetalTiger:
----------------------

./MetalTiger transform -config metaltiger-master/sample_models/lm_clean/en_US/spokenform.toml -input all_speech.txt \
        > output-all_speech.txt


Keyshortcut:
---------------
Up key: loads previous command
htop: shows bolts task manager
