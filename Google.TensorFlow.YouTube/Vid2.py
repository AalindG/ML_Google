import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sent = [
    'This is sentence one',
    'This is sentence two',
    'That is sentence! three',
    'Do you believe this sentence is amazing?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sent)
word_index = tokenizer.word_index
initial_text_to_sequence = tokenizer.texts_to_sequences(sent)

padded = pad_sequences(initial_text_to_sequence)
# padded = pad_sequences(initial_text_to_sequence,
#                        padding='post', truncating='post', maxlen=6)

test_data = [
    'This is a new sentence',
    'This new sentence is in English'
]

print(word_index)
print(initial_text_to_sequence)
print(padded)

test_text_to_sequence = tokenizer.texts_to_sequences(test_data)

print(test_text_to_sequence)
