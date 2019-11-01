from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import keras.utils as ku 
import numpy as np
import json
import jellyfish

# Load lyrics from json file
def load_json_data(data):
    with open(data, 'r') as f:
        corpus = json.load(f)
        return corpus

# Data preprocessing       
def data_prep(corpus):
    for song in corpus:
        tokenizer.fit_on_texts(set(song['lyric']))
    # Convert each line to a numeric sequence
    input_seq = []
    for song in corpus:
        song_seq = tokenizer.texts_to_sequences(song['lyric'])
        # Collect n-grams
        for line in song_seq:
            for i in range(1, len(line)):
                n_gram = line[:i+1]
                input_seq.append(n_gram)
    
    # Pad all sequences to be equal to the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_seq])
    input_seq = np.array(pad_sequences(input_seq,   
                          maxlen=max_sequence_len, padding='pre'))

    # Labels are the last word of each sequence             
    predictors, label = input_seq[:,:-1],input_seq[:,-1]
    # Convert labels to one-hot categorical representations
    label = ku.to_categorical(label, num_classes=len(tokenizer.word_index)+1)

    return predictors, label, max_sequence_len, len(tokenizer.word_index) + 1


tokenizer = Tokenizer()

X, y, max_seq_len, total_words = data_prep(load_json_data('rh_lyrics.json'))

# Create model
def get_model(X = X, y =y, max_seq_len = max_seq_len, total_words= total_words):
    model = Sequential()
    # Input_dim = number of unique words in the vocabulary, output_dim = embedding dimension, input_length = sequence length
    model.add(Embedding(input_dim = total_words, output_dim = 32, input_length = max_seq_len-1))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    history = model.fit(X, y, epochs = 50, batch_size=32)
    model.save('rh_lang_model.h5')
    return model

# Generate text from given seed text where next_words is an integer 
def generate_text(seed_text, num_lines, path_to_model, max_sequence_len):
    model = load_model(path_to_model)
    for line in range(num_lines):
        delim = np.random.randint(4, 9)
        for _ in range(delim):
            # Convert seed text to a sequence
            seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
            # Pad to match max_seq_len
            seed_seq = pad_sequences([seed_seq], maxlen=max_sequence_len -1, padding='pre')
            # Predict next word given the sequence
            predicted = model.predict_classes(seed_seq, verbose=0)

            output_word = ""
            # Match index to corresponding word
            for word,index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            # Add the output to the input seed text
            seed_text += " "+output_word
        if line == 0:
            print(seed_text)
        else:
            tok_seed = seed_text.split()
            # Pass the generated text as new input
            seed_text = ' '.join(tok_seed[-delim:])
            print(seed_text)




