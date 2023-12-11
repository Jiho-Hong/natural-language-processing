# Transformer 모델을 이용한 단위 기계 번역(영어 - 스페인어)
import urllib.request
import zipfile
import numpy as np
import keras_core as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder

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

# 영어, 스페인어 문장을 나누어 저장
en_texts, es_texts = [], []
for line in lines:
    input_text, target_text = line.split('\t')

    # 영어
    en_texts.append(input_text)
    # 스페인어
    es_texts.append('[START] ' + target_text + ' [END]')

# 데이터 길이
encoder_max_len = max(len(line.split()) for line in en_texts)
decoder_max_len = max(len(line.split()) for line in es_texts)
seq_len = max(encoder_max_len, decoder_max_len)

# # 데이터 확인
# import random
#
# random_idx = random.choice(range(len(lines)))
# print(lines[random_idx])
# print(en_texts[random_idx])
# print(es_texts[random_idx])

# 정수 인코딩
# 토크나이저 생성
tokenizer_en = Tokenizer(filters="", lower=False)
tokenizer_es = Tokenizer(filters="", lower=False)

# 빈도수 기준으로 단어 set 생성
tokenizer_en.fit_on_texts(en_texts)
tokenizer_es.fit_on_texts(es_texts)

# 인덱스로 변환
en_x = tokenizer_en.texts_to_sequences(en_texts)
es_y = tokenizer_es.texts_to_sequences(es_texts)

# 패딩 추가
en_x = pad_sequences(en_x, maxlen=seq_len, padding="post")
es_y = pad_sequences(es_y, maxlen=seq_len + 1, padding="post")

# Vocab 크기
en_vocab_size = len(tokenizer_en.word_index) + 1
es_vocab_size = len(tokenizer_es.word_index) + 1

encoder_inputs = en_x
decoder_inputs = es_y[:, :-1]
decoder_outputs = es_y[:, 1:]

num_head = 8
embedding_size = 256
# 인코더
encoder_input = keras.Input(shape=(None,), dtype='int64')
x = TokenAndPositionEmbedding(en_vocab_size, seq_len, embedding_size)(encoder_input)
encoder_output = TransformerEncoder(embedding_size, num_head)(x)
encoded_seq_input = keras.Input(shape=(None, embedding_size))

# 디코더
decoder_input = keras.Input(shape=(None,), dtype='int64')
x = TokenAndPositionEmbedding(es_vocab_size, seq_len, embedding_size, mask_zero=True)(decoder_input)
x = TransformerDecoder(embedding_size, num_head)(x, encoded_seq_input)
x = keras.layers.Dropout(0.3)(x)

decoder_output = keras.layers.Dense(es_vocab_size, activation='softmax')(x)
decoder = keras.Model([decoder_input, encoded_seq_input], decoder_output)
decoder_output = decoder([decoder_input, encoder_output])

# 모델 생성
model = keras.Model([encoder_input, decoder_input], decoder_output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# 학습
model.fit([encoder_inputs, decoder_inputs], decoder_outputs, epochs=50, validation_split=0.2)

# 기계 번역 실행
decoder_idx_to_word = dict(zip(range(len(tokenizer_es.word_index)), tokenizer_es.word_index))


def decode_sequence(text):
    input_seq = tokenizer_en.texts_to_sequences([text])
    input_seq = pad_sequences(input_seq, maxlen=seq_len, padding='post')

    seq_idx = 0
    decoded_text = '[START]'
    while True:
        target_seq = tokenizer_es.texts_to_sequences([decoded_text])
        target_seq = pad_sequences(target_seq, maxlen=seq_len, padding='post')

        prediction = model([input_seq, target_seq])

        word_idx = np.argmax(prediction[0, seq_idx, :]) - 1
        token = decoder_idx_to_word[word_idx]
        decoded_text += ' ' + token

        seq_idx += 1

        if token == '[END]' or seq_idx >= seq_len:
            break

    return decoded_text[len('[START] '):-len(' [END]')]


for sequence_idx in np.random.choice(len(lines), 10):
    input_sentence = en_texts[sequence_idx]
    decoded_sentence = decode_sequence(input_sentence)

    print('Translation result')
    print(f"Input sentence: {input_sentence}")
    print(f"Decoded sentence: {decoded_sentence}\n\n")
