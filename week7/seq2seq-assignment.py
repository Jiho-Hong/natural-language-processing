# seq2seq 모델을 이용한 단어 단위 기계 번역(영어 - 스페인어)
import urllib.request
import zipfile
import string

import keras.optimizers
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 데이터 불러오기
# 출처: http://www.manythings.org/
url = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'
urllib.request.urlretrieve(url=url, filename='spa-eng.zip')
zipfile.ZipFile('spa-eng.zip').extractall()

data_file = 'spa-eng/spa.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')[:-1]

# 총 118,964개 데이터 중 10,000개만 사용
lines = lines[:10000]


def preprocessing(text):
    text = text.lower()
    text_list = []
    for char in text:
        if char not in string.punctuation:
            text_list.append(char)

    return ''.join(text_list)


# 영어(입력), 스페인어(타깃) 문장을 나누어 저장
# 타깃 문장 앞에 'start sequence' 단어로 '[START]', 뒤에는 'end sequence' 단어로 '[END]' 추가
input_target_texts = []
for line in lines:
    input_text, target_text = line.split('\t')

    input_text = preprocessing(input_text)

    target_text = preprocessing(target_text)
    target_text = '[START]' + ' ' + target_text + ' ' + '[END]'

    input_target_texts.append((input_text, target_text))

# 데이터 확인
# import random
#
# random_idx = random.choice(range(len(lines)))
# print(lines[random_idx])
# print(input_target_texts[random_idx])

# 입력, 타깃 문장에 사용된 단어를 set으로 저장 후 정렬
input_words, target_words = set(), set()
for input_text, target_text in input_target_texts:
    input_words.update(input_text.split())
    target_words.update(target_text.split())

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))

# 인코더, 디코더 입력과 디코더 타깃(실제값) 저장
max_encoder_seq_len = max(len(input_text.split()) for input_text, _ in input_target_texts)
max_decoder_seq_len = max(len(target_text.split()) for _, target_text in input_target_texts)

encoder_input_data = np.zeros(shape=(len(input_target_texts), max_encoder_seq_len), dtype='float32')
decoder_input_data = np.zeros(shape=(len(input_target_texts), max_decoder_seq_len), dtype='float32')
decoder_target_data = np.zeros(shape=(len(input_target_texts), max_decoder_seq_len, len(target_words)), dtype='float32')

# 각 단어에 인덱스 부여
input_word_to_idx = dict([(word, i) for i, word in enumerate(input_words)])
target_word_to_idx = dict([(word, i) for i, word in enumerate(target_words)])

# 원-핫 인코딩
for i, (input_text, target_text) in enumerate(input_target_texts):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_word_to_idx[word]

    for t, word in enumerate(target_text.split()):
        decoder_input_data[i, t] = target_word_to_idx[word]
        if t > 0:
            decoder_target_data[i, t - 1, target_word_to_idx[word]] = 1.0

latent_dim = 256
embedding_size = 32
# 인코더
encoder_inputs = Input(shape=(None,))
embedded_encoder_inputs = Embedding(len(input_words), embedding_size, mask_zero=True)(encoder_inputs)
encoder = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embedded_encoder_inputs)

encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(None,))
embedded_decoder_inputs = Embedding(len(target_words), embedding_size, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embedded_decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(units=len(target_words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 생성
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01), loss='categorical_crossentropy',
              metrics='accuracy')
print(model.summary())

# 학습
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=10, validation_split=0.2)

# 기계 번역 실행
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h, decoder_state_input_c = Input(shape=(latent_dim,)), Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

embedded_decoder_inputs = Embedding(len(target_words), embedding_size, mask_zero=True)(decoder_inputs)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(embedded_decoder_inputs,
                                                                 initial_state=decoder_states_inputs)
decoder_states = [decoder_state_h, decoder_state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

input_idx_to_word = {idx: char for char, idx in input_word_to_idx.items()}
target_idx_to_word = {idx: char for char, idx in target_word_to_idx.items()}


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_to_idx['[START]']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_idx_to_word[sampled_token_idx]
        decoded_sentence += ' ' + sampled_word

        if sampled_word == '[END]' or len(decoded_sentence) > max_decoder_seq_len:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_idx

        states_value = [h, c]

    end_sequence = ' ' + '[END]'
    if decoded_sentence[-len(end_sequence):] == end_sequence:
        decoded_sentence = decoded_sentence[1:-len(end_sequence)]

    return decoded_sentence


for seq_idx in np.random.choice(len(lines), 10):
    input_seq = encoder_input_data[seq_idx:seq_idx + 1]
    decoded_sentence = decode_sequence(input_seq)

    print('Translation result')
    print(f"Input sentence: {input_target_texts[seq_idx][0]}")
    print(f"Decoded sentence: {decoded_sentence}\n\n")
