# Reason behind V 2.0
For our model we need handle both X->Y and Y->X.
Which is very difficult for text normalizer as X->Y and Y->X.
In version 1.0 we trianed with text with context and speech without context directly. 
This cause cycle gan to give poor performance on Y->X.
We are trying to solve this issue in V2.0
