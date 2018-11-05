import tensorflow as tf
import keras
import numpy as np

print(tf.__version__)

# loading train data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

#Explore the data
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
'''The text of reviews have been converted to integers, 
where each integer represents a specific word in a dictionary. 
Here's what the first review looks like:
And how did they convert it into numbers?
'''
print(train_data[0])
'''Movie reviews may be different lengths. 
The below code shows the number of words in the first and 
second reviews. Since inputs to a neural network must be 
the same length, we'll need to resolve this later.
'''
print(len(train_data[0]), len(train_data[1]))
'''
The dictionary is used to convert the text to data
'''
# A dictionary mapping words to an integer index
# There is a index for the words for the 
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

#Prepare the data, they need to be tokenized
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#Review the data 
print(len(train_data[0]), len(train_data[1]))


