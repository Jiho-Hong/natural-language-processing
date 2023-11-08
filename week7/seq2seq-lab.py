# seq2seq 모델을 이용한 문자 단위 기계 번역(영어 - 스페인어)
import urllib.request
import zipfile
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# 데이터 불러오기
# 출처: http://www.manythings.org/
url = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'
urllib.request.urlretrieve(url=url, filename='spa-eng.zip')
zipfile.ZipFile('spa-eng.zip').extractall()

data_file = 'spa-eng/spa.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')[:-1]

# 총 118,964개 데이터 중 30,000개만 사용
lines = lines[:30000]

# 영어(입력), 스페인어(타깃) 문장을 나누어 저장
# 타깃 문장 앞에 'start sequence' 문자로 탭('\t'), 뒤에는 'end sequence' 문자로 줄 바꿈('\n') 추가
input_target_texts = []
for line in lines:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_target_texts.append((input_text, target_text))

# 데이터 확인
# import random
#
# random_idx = random.choice(range(len(lines)))
# print(lines[random_idx])
# print(input_target_texts[random_idx])

# 입력, 타깃 문장에 사용된 문자를 set으로 저장 후 정렬
input_chars, target_chars = set(), set()
for input_text, target_text in input_target_texts:
    input_chars.update(input_text)
    target_chars.update(target_text)

input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))

# 인코더, 디코더 입력과 디코더 타깃(실제값) 저장
max_encoder_seq_len = max(len(input_text) for input_text, _ in input_target_texts)
max_decoder_seq_len = max(len(target_text) for _, target_text in input_target_texts)

encoder_input_data = np.zeros(shape=(len(input_target_texts), max_encoder_seq_len, len(input_chars)), dtype='float32')
decoder_input_data = np.zeros(shape=(len(input_target_texts), max_decoder_seq_len, len(target_chars)), dtype='float32')
decoder_target_data = np.zeros(shape=(len(input_target_texts), max_decoder_seq_len, len(target_chars)), dtype='float32')

# 각 문자에 인덱스 부여
input_char_to_idx = dict([(char, i) for i, char in enumerate(input_chars)])
target_char_to_idx = dict([(char, i) for i, char in enumerate(target_chars)])

# 원-핫 인코딩
for i, (input_text, target_text) in enumerate(input_target_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_char_to_idx[char]] = 1.0
    encoder_input_data[i, t + 1:, input_char_to_idx[" "]] = 1.0

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_char_to_idx[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_char_to_idx[char]] = 1.0
    decoder_input_data[i, t + 1:, target_char_to_idx[" "]] = 1.0
    decoder_target_data[i, t:, target_char_to_idx[" "]] = 1.0

latent_dim = 64
# 인코더
encoder_inputs = Input(shape=(None, len(input_chars)))
encoder = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(None, len(target_chars)))

decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(units=len(target_chars), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 생성
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')
print(model.summary())

# 학습
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=10, validation_split=0.2)

# 기계 번역 실행
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h, decoder_state_input_c = Input(shape=(latent_dim,)), Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [decoder_state_h, decoder_state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

input_idx_to_char = {idx: char for char, idx in input_char_to_idx.items()}
target_idx_to_char = {idx: char for char, idx in target_char_to_idx.items()}


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, len(target_chars)))
    target_seq[0, 0, target_char_to_idx['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_idx_to_char[sampled_token_idx]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_len:
            stop_condition = True

        target_seq = np.zeros((1, 1, len(target_chars)))
        target_seq[0, 0, sampled_token_idx] = 1.0

        states_value = [h, c]

    return decoded_sentence


for seq_idx in np.random.choice(len(lines), 10):
    input_seq = encoder_input_data[seq_idx:seq_idx + 1]
    decoded_sentence = decode_sequence(input_seq)

    print('Translation result')
    print(f"Input sentence: {input_target_texts[seq_idx][0]}")
    print(f"Decoded sentence: {decoded_sentence}\n\n")
