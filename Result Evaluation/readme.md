Implemented different machine translation algorithm and model then evaluated them. 
Seqence is as follows:

1. simple encoder decoder based seq2seq model with bidirectional lstm layer (evaluated with and without context)
2. Attention Mechanism added to the 1 model and evaluated using context and without context
3. Applied GAN to observe adversarial loss effect on generator performance which is model of 1 and 2.  (evaluated with winning data from 1 & 2)
4. Applied Cycle GAN that handle two generator normalization and inverse text normalization at the same time and evaluate the model
