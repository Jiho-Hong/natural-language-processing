# Word2Vec CBOW를 이용해 임베딩 학습하기
from konlpy.tag import Okt
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Lambda, Dense
from keras import backend

# 훈련 데이터
train_data = [
    '나는 고양이를 정말 좋아해요',
    '나는 강아지를 정말 싫어하지 않아요',
    '고양이는 너무 귀엽습니다',
    '강아지는 너무 똑똑하죠',
    '동물들은 정말로 다양하다고 생각해요',
    '나는 동물을 아주 사랑합니다',
    '나는 고양이와 강아지를 함께 키우고 있어요',
    '친구가 고양이를 아주 좋아하고 있어요',
    '동물 병원에 자주 갔었어요',
    '나는 동물을 매일 사랑하고 있어요'
]

tokenizer = Okt()

train_tokenized = [tokenizer.morphs(sent, stem=True) for sent in train_data]
print(train_tokenized)

i2w = set(chain.from_iterable(train_tokenized))
print(i2w)
w2i = {w: i for i, w in enumerate(i2w)}
print(w2i)

vocab_size = len(i2w)
embedding_size = 5
window_size = 2


def preprocecssing(train_tokenized):
    context, target = [], []
    for tokens in train_tokenized:
        token_ids = [w2i[token] for token in tokens]
        for i, id in enumerate(token_ids):
            if i - window_size >= 0 and i + window_size < len(token_ids):
                context.append(token_ids[i - window_size:i] + token_ids[i + 1:i + window_size + 1])
                target.append(token_ids[i])

    context = pad_sequences(context, maxlen=2 * window_size, padding='post')
    target = to_categorical(target, num_classes=vocab_size)

    return context, target


# 데이터 전처리
context, target = preprocecssing(train_tokenized)

# 모델 생성
cbow_model = Sequential()

cbow_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=window_size * 2))
cbow_model.add(Lambda(lambda x: backend.mean(x, axis=1), output_shape=embedding_size))
cbow_model.add(Dense(vocab_size, activation='softmax'))

cbow_model.compile(optimizer='sgd', loss='categorical_crossentropy')
print(cbow_model.summary())

# 학습
cbow_model.fit(context, target, batch_size=4, epochs=100, verbose=1)

# 테스트 데이터
test_words = ['동물', '강아지', '고양이', '사랑']

# CBOW 임베딩 확인
print('CBOW 임베딩')
weights = cbow_model.get_weights()
embedding = weights[0]
for word in test_words:
    word_idx = w2i[word]
    word_embedding = embedding[word_idx]

    print(f'{word} embedding: {word_embedding}')
