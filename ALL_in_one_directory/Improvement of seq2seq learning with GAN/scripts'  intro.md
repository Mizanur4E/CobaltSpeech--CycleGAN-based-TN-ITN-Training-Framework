
**enc-dec based seq2seq (baseline model).py** --
implementation of encoder-decoder based sequence to sequence conversion model. Encoder and Decoder are LSTM network.

**enc-dec based bi-lstm seq2seq.py** --
Used Bi-directional LSTM instead of LSTM in both encoder and decoder.

**Dual Embedding Issue Regeneration GAN.py** --
In simple GAN, generator and discriminator added together for combined model, so, discriminator 
embedding layer comes in between the network. This causes failure of gradient backpropagation.

**alternateEmbeddingSeq2seq.py** --
Dense layer is used as an alternative of embedding layer. Tested on a seq2seq model in this script.

**WasserteinGANforText.py** --
Wassertein Loss is applied for proper gradient generation from discriminator. 

**WGAN-GP-GAN.py** --
'Improved WGAN-training with gradient penalty'-has been implemented for seq2seq network.

**Simple GAN with custom word2vec.py** --
To make sequences continuous they are converted to embedding vec first using word2vec. This script
implements the network.

**CycleGAN in single Graph.py**-- 
whole CycleGAN can be trained under a single graph. Using Model subclassing it is shown in this script.
This speed up the training.

**CycleGANwithContext_text2text.py**--
CycleGAN has been implemented where Gxy -> generates y (normalized text) with context given in x,
Gyx -> generates x (formatted text) without any context given in y.

**WGAN-GP-CycleGAN_vect2vect.py** --
instead of seq2seq conversion in cycleGAN, vect2vect was tried to implement.

**AdversarialLossWtihSeq2seqLoss.py** --
Trains enc-dec based seq2seq model with categoricalCrossentropy loss and adversarial loss
