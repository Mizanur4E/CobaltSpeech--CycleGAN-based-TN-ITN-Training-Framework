# Text-normalizer-formatter-CobaltSpeech

------------Commands for communication--------

ssh nayan@bolt.in.cobaltspeech.com      
jupyter notebook --no-browser --port=1234

ssh -L 8080:localhost:1234 nayan@100.88.210.1
http://localhost:8080/



---------------------Readme:----------------------

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

scp classwiseoutputinputv2.csv nayan@100.88.210.1:/home/nayan

scp espresso_tt.py nayan@100.88.210.1:/home/nayan




For running any script:

nvidia-smi          #checking GPU

LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" PATH="/usr/local/cuda-11.2/bin:$PATH" CUDA_VISIBLE_DEVICES="1" python3 espresso_tt.py

