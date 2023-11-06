# LSTM을 이용한 문장 생성 모델 만들기 - KoNLPy 한국 법률 말뭉치 활용
import numpy as np

from konlpy.corpus import kolaw
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

kolaw_text = kolaw.open('constitution.txt').read()
sentences = [s for s in kolaw_text.split('\n') if s]

# 위 문장을 이용하여 단어와 인덱스로 이루어진 딕셔너리 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
print(f'단어 인덱스: {tokenizer.word_index}')

# 훈련 데이터 생성
x_train, y_train = [], []
for sentence in sentences:
    sequence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(sequence)):
        x_train.append(sequence[:i])
        y_train.append(sequence[i])

x_train, y_train = np.array(x_train, dtype='object'), np.array(y_train, dtype='object')

# 패딩 추가
max_data_len = max(len(e) for e in x_train)
x_train = pad_sequences(x_train, maxlen=max_data_len, padding='pre')

# 원-핫 인코딩
vocab_size = len(tokenizer.word_index) + 1
y_train = to_categorical(y_train, num_classes=vocab_size)

# 훈련 데이터 확인
# num_data = 10
# for x, y in zip(x_train[:num_data], y_train[:num_data]):
#     print(x, y)

# 모델 생성
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=32))
model.add(LSTM(units=128))  # hidden size: 128
model.add(Dense(units=vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary(()))

# 학습
model.fit(x_train, y_train, batch_size=16, epochs=30, verbose=1)

# 문장 생성
word = '법률'
num_words = 20

gen_sentence = word
for _ in range(num_words):
    gen_sequences = tokenizer.texts_to_sequences([gen_sentence])[0]
    gen_sequences = pad_sequences([gen_sequences], maxlen=max_data_len, padding='pre')

    output = model.predict(gen_sequences, verbose=0)
    output = np.argmax(output, axis=1)

    gen_sentence += ' ' + list(tokenizer.word_index.items())[output[0]][0]

print(f'생성된 문장: {gen_sentence}')
