# CNN을 이용하여 네이버 영화 리뷰 데이터 분류(긍정 혹은 부정)하기
import urllib.request
import numpy as np
from konlpy.tag import Okt
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense

# 네이버 영화 리뷰 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                           filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                           filename="ratings_test.txt")

# 다운로드한 파일 확인
f_ratings_train = open('ratings_train.txt', 'r')
train_lines = f_ratings_train.readlines()
f_ratings_train.close()

f_ratings_test = open('ratings_test.txt', 'r')
test_lines = f_ratings_test.readlines()
f_ratings_test.close()

# 5번째 라인까지 출력
# 탭 문자 기준으로 구분
# 첫 번째 라인에는 열의 이름 저장(id, document, label)
for i in range(5):
    print(train_lines[i])

# 훈련 데이터 20,000개, 테스트 데이터 4,000개
num_train_data, num_test_data = 20000, 4000


def preprocessing(train_lines, test_lines):
    # 학습에 사용하지 않을 'id'열을 제외
    x_train = [line.split('\t')[1].strip() for line in train_lines[1:num_train_data + 1]]
    y_train = [int(line.split('\t')[2].strip()) for line in train_lines[1:num_train_data + 1]]

    x_test = [line.split('\t')[1].strip() for line in test_lines[1:num_test_data + 1]]
    y_test = [int(line.split('\t')[2].strip()) for line in test_lines[1:num_test_data + 1]]

    # 문장 토크나이징
    tokenizer = Okt()

    x_train = [tokenizer.morphs(sentence) for sentence in x_train]
    x_test = [tokenizer.morphs(sentence) for sentence in x_test]

    # 모든 단어 저장
    words = []
    for tokens in x_train:
        words.extend(tokens)

    # 각 단어의 등장 빈도가 높으면 낮은 숫자로 표현하기 위해 Counter 사용
    x_train_counter = Counter(words)
    x_train_counter = x_train_counter.most_common()

    vocab = ['<PADDING>', '<START>', '<UNKNOWN>'] + [key for key, value in x_train_counter]
    w2i = {word: idx for idx, word in enumerate(vocab)}

    def token_to_index(tokens):
        return [w2i[token] if token in w2i else w2i['<UNKNOWN>'] for token in tokens]

    # 각 데이터의 단어를 인덱스로 변환
    x_train = [token_to_index(tokens) for tokens in x_train]
    x_test = [token_to_index(tokens) for tokens in x_test]

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test)), vocab, w2i


(x_train, y_train), (x_test, y_test), i2w, w2i = preprocessing(train_lines=train_lines, test_lines=test_lines)

# 최대 max_len값만큼 패딩 추가
max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 모델 생성
model = Sequential()

vocab_size = len(i2w)
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
