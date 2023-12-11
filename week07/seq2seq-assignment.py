# seq2seq 모델을 이용한 단어 단위 기계 번역(영어 - 스페인어)
import urllib.request
import zipfile
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Masking, LSTM, Dense
from keras.models import Model

# 데이터 불러오기
# 출처: http://www.manythings.org/
url = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'
urllib.request.urlretrieve(url=url, filename='spa-eng.zip')
zipfile.ZipFile('spa-eng.zip').extractall()

data_file = 'spa-eng/spa.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')[:-1]

# 데이터 셔플
indices = np.arange(len(lines))
np.random.shuffle(indices)

# 총 118,964개 데이터 중 50,000개만 사용
lines = [lines[indices[i]] for i in range(50000)]

# 영어(인코더), 스페인어(디코더) 문장을 나누어 저장
encoder_input_texts, decoder_input_texts, decoder_target_texts = [], [], []
for line in lines:
    input_text, target_text = line.split('\t')

    # 인코더 input
    encoder_input_texts.append(input_text)
    # 디코더 input
    decoder_input_texts.append('[START] ' + target_text)
    # 디코더 target
    decoder_target_texts.append(target_text + ' [END]')

# 데이터 확인
# import random
#
# random_idx = random.choice(range(len(lines)))
# print(lines[random_idx])
# print(encoder_input_texts[random_idx])
# print(decoder_input_texts[random_idx])
# print(decoder_target_texts[random_idx])

# 정수 인코딩
# 토크나이저 생성
tokenizer_encoder = Tokenizer(filters="", lower=False)
tokenizer_decoder = Tokenizer(filters="", lower=False)

# 빈도수 기준으로 단어 set 생성
tokenizer_encoder.fit_on_texts(encoder_input_texts)
tokenizer_decoder.fit_on_texts(decoder_input_texts)
tokenizer_decoder.fit_on_texts(decoder_target_texts)

# 인덱스로 변환
encoder_input = tokenizer_encoder.texts_to_sequences(encoder_input_texts)
decoder_input = tokenizer_decoder.texts_to_sequences(decoder_input_texts)
decoder_target = tokenizer_decoder.texts_to_sequences(decoder_target_texts)

# 패딩 추가
encoder_input = pad_sequences(encoder_input, padding="post")
decoder_input = pad_sequences(decoder_input, padding="post")
decoder_target = pad_sequences(decoder_target, padding="post")

latent_dim = 64
embedding_size = 64
# 인코더
encoder_inputs = Input(shape=(None,))
embedding_encoder_inputs = Embedding(len(tokenizer_encoder.word_index), embedding_size, mask_zero=True)(encoder_inputs)
embedding_encoder_inputs = Masking(mask_value=0.0)(embedding_encoder_inputs)
encoder = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embedding_encoder_inputs)

encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(None,))
embedding_layer = Embedding(len(tokenizer_decoder.word_index), embedding_size, mask_zero=True)
embedding_decoder_inputs = embedding_layer(decoder_inputs)
embedding_decoder_inputs = Masking(mask_value=0.0)(embedding_decoder_inputs)
decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embedding_decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(units=len(tokenizer_decoder.word_index), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 생성
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
print(model.summary())

# 학습
model.fit([encoder_input, decoder_input], decoder_target, batch_size=256, epochs=10, validation_split=0.1)

# 기계 번역 실행
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h, decoder_state_input_c = Input(shape=(latent_dim,)), Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

embedded_decoder_inputs = embedding_layer(decoder_inputs)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(embedded_decoder_inputs,
                                                                 initial_state=decoder_states_inputs)
decoder_states = [decoder_state_h, decoder_state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

decoder_word_to_idx = tokenizer_decoder.word_index
decoder_idx_to_word = tokenizer_decoder.index_word

max_decoder_seq_len = max(len(decoder_input_text) for decoder_input_text in decoder_input_texts)


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = decoder_word_to_idx['[START]']

    stop_condition = False
    decoded_words = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = decoder_idx_to_word[sampled_token_idx]
        decoded_words.append(sampled_word)

        if sampled_word == '[END]' or len(decoded_words) > max_decoder_seq_len:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_idx

        states_value = [h, c]

    if decoded_words[-1] == '[END]':
        decoded_words = decoded_words[:-1]

    return ' '.join(decoded_words)


for seq_idx in np.random.choice(len(lines), 10):
    input_seq = encoder_input[seq_idx:seq_idx + 1]
    decoded_sentence = decode_sequence(input_seq)

    print('Translation result')
    print(f"Input sentence: {encoder_input_texts[seq_idx]}")
    print(f"Decoded sentence: {decoded_sentence}\n\n")
