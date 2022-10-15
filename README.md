# Statistical text-normalizer and formatter-CobaltSpeech

ALL_in_one_directory
-----------------------
The directory contains final scripts for data preprocessing, model building, training and evaluation. 

Formatter
---------------
This directory contains script for processing voiceScript data. 'FormatterDataPreprocessorForMT.py' to generate speaker speech pair.
It also contains python script to yield dataset for statistical formatter trainning. We used metal tiger to generate formatted form of voiceScript data. Then we used sclite to produce comparsion files. From this comparison files we produced formatted form of the text and speech form. 'Applying MetalTiger and sclite for dataset preparation.ipynb'
We also produced IOB tagger's dataset in the notebook.

Result Evaluation:
-----------------------

This directory contains python scripts to generate results from different statistical text normalization and formatting approach. The directory contains a readme.md file that explains the contained files in detalls.


Scripts
-----------
It contains script to preprocess google dataset 'data-preprocessing.py' and implement cycle GAN 'espresso.py'. 'load_and_predicts.py' contains python scripts to evaluate text normalizer for all 14 semiotic class.
V 2.0  directory in this directory is a modified version of cycle gan model. where model is evaluated by measuring WER, BLEU and Accuracy metric. In V 2.0 we solved poor performance of Y->X conversion which was generated due to context in the X. (here X means Not normalized text and Y means normalized text). It is suggested to work with code of V 2.0. 

Dataset links
------------
It contains the link of the google dataset used to train and evaluate the model

Demo_CoW_21_3_Statistical_Formatter.ipynb
----------------------------
This notebook contains demo version of our proposed statistical formatter, where we implemented IOB tagger model and text formatter model. Then we concatenated two models output to generate formatted text from input sentences. 


Bolt login and necessary commands
----------------------------------

I found this commands and instructions very handy to communicate with bolt. File transfering, gpu availability checking, python scripting running etc needs some basic commands. I tried to list the most common instructions in this file.
