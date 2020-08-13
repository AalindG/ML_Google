import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

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


print(word_index)
print(initial_text_to_sequence)
