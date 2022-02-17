**VoiceScript_data_processing directory** contains python scripts for data preparation. A colab notebook has also been added 
if anyone wants interactive run from jupyter notebook.

Processed data can be used for training of formatter, normalizer and IOB tagger model. Speaker-speech pair data
can be used for other NLP task like ASR with speaker detection.

**Statistical_formatter_demo.ipynb** this notebook demonstrates how to connect two trained model to build formatting pipeline. 
IOB tagger model and formatter model are connected together. Tagger model detect the location of 
text to be formatted. With this info data is extracted and feed to formatter. Formatter output then 
replaced in the main sentence with speech form text. 
