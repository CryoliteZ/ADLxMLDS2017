import numpy as np
import pandas as pd
import os, sys
import pickle
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential, load_model, model_from_json
from keras.models import model_from_json



def decode_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, tokenizer.word_index['bos']] = 1

    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            sampled_char = ''
        else:
            sampled_char = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(sampled_token_index)]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def getX_test(path):   
    X_test = []
    X_test_filename = []
    for filename in os.listdir(path):
        f = np.load(os.path.join(path , filename))
        X_test.append(f)
        X_test_filename.append(filename[:-4])
    X_test = np.array(X_test)
    return X_test, X_test_filename

data_directory = sys.argv[1]
output_file = sys.argv[2]

num_encoder_tokens = 4096
num_decoder_tokens = 1024
latent_dim = 512


with open('data/tokinzer1024', 'rb') as file:
    tokenizer = joblib.load(file)
    print(len(tokenizer.word_index))

enc_model = load_model('encoder_model.h5')
enc_model.summary()
dec_model = decode_model(num_encoder_tokens, num_decoder_tokens, latent_dim)
dec_model.load_weights('decoder_model_weights.h5')
dec_model.summary()


X_test, X_test_filename = getX_test(os.path.join(data_directory, "testing_data/feat/"))

specail_vid = ["klteYv1Uv9A_27_33.avi", "5YJaS2Eswg0_22_26.avi", "UbmZAe5u5FI_132_141.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]

with open(output_file, 'w') as file:
    for idx, x in enumerate(X_test): 
        if(X_test_filename[idx] not in specail_vid):
            continue
        decoded_sentence = decode_sequence(x.reshape(-1, 80, 4096), enc_model,dec_model)
        decode_str = ''
        filter_string = ['bos', 'eos']
        for c in decoded_sentence:
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str += c + ' '
        print(X_test_filename[idx] + ' > ' + decode_str)
        file.write(X_test_filename[idx] + ',' + decode_str + '\n')
