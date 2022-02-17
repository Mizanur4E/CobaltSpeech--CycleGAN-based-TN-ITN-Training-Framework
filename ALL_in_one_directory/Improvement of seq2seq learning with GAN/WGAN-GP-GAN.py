'''
Implementation of WGAN with gradient penalty. 
Generator has normal embedding layer. Output is softmax vector of vocab_size. Discriminator has
generator input as context and generator output(softmax output)  as input. Discriminator is trained using 
loss described in WGAN-GP paper.Generator is trained using classical_crossentropy and false_y_sample score.
'''
