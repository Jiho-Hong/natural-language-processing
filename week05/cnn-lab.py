# CNN을 이용하여 IMDB 영화 리뷰 데이터 분류(긍정 혹은 부정)하기
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense

# Keras 데이터 세트의 IMDB 데이터

# 빈도 순으로 상위 10000개 단어만 사용
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 데이터 개수
print(f"훈련 데이터 개수: {len(x_train)}")
print(f"테스트 데이터 개수: {len(x_test)}")

# 리뷰 데이터 구성
# 각 단어의 등장 빈도가 높으면 낮은 숫자로 표현
print(f"첫 번째 리뷰 데이터: {x_train[0]}")

# 숫자로 표현된 데이터의 변환전 단어 확인
w2i = imdb.get_word_index()
i2w = {}

i2w[0] = "<PADDING>"
i2w[1] = "<START>"
i2w[2] = "<UNKNOWN>"

for key, value in w2i.items():
    i2w[value + 3] = key

print(f"첫 번째 리뷰 데이터의 변환 전 단어들\n{' '.join([i2w[e] for e in x_train[0]])}")

# 리뷰 레이블 데이터 구성
# 긍정: 1, 부정: 0
print(f"리뷰 레이블 데이터 종류: {set(y_train)}")

# 최대 max_len값만큼 패딩 추가
max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 모델 생성
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
model.add(Dropout(rate=0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary(()))

# 학습
model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(x_test, y_test))

# 평가
loss, acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {acc * 100:.2f}%")
