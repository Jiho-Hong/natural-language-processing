# RNN을 이용한 문장 생성 모델 만들기
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

sentences = [
    "오늘은 맑은 날씨에 햇볕이 좀 덥게 내리쬐고 있어.",
    "가까운 미래에 자율 주행 자동차가 보편화될 것으로 예상한다.",
    "요즘은 온라인 강의를 통해 새로운 기술을 배우기가 훨씬 수월해졌다.",
    "한국의 봄은 벚꽃이 만개하여 정말 아름답다.",
    "음악을 듣는 것은 스트레스를 해소하는 좋은 방법 중에 하나이다.",
    "올해는 여름휴가로 제주도에 가보려고 계획 중이다.",
    "과학 기술의 발전으로 우리의 삶이 훨씬 더 편리해졌다.",
    "요즘은 건강을 위해 꾸준한 운동이 중요하게 강조된다.",
    "한국의 전통 음식 중에서도 불고기가 제일 맛있는 것 같아.",
    "좋은 책을 읽는 것은 마음을 풍부하게 하는 데 큰 도움이 된다."
]

# 위 문장을 이용하여 단어와 인덱스로 이루어진 딕셔너리 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)

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
model.add(SimpleRNN(units=64))  # hidden size: 64
model.add(Dense(units=vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary(()))

# 학습
model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1)

# 문장 생성
word = '한국의'
num_words = 5

gen_sentence = word
for _ in range(num_words):
    gen_sequences = tokenizer.texts_to_sequences([gen_sentence])[0]
    gen_sequences = pad_sequences([gen_sequences], maxlen=max_data_len, padding='pre')

    output = model.predict(gen_sequences, verbose=0)
    output = np.argmax(output, axis=1)

    gen_sentence += ' ' + list(tokenizer.word_index.items())[output[0]][0]

print(gen_sentence)
