# 임베딩을 이용해 단어 간 유사도 확인하기
from gensim.models import Word2Vec

# 텍스트 데이터
sentences = [
    ["The", "cat", "is", "on", "the", "mat"],
    ["He", "plays", "the", "guitar", "every", "day"],
    ["She", "enjoys", "reading", "books", "in", "her", "free", "time"],
    ["They", "went", "to", "the", "beach", "for", "a", "picnic"],
    ["I", "have", "a", "meeting", "at", "3", "o'clock"],
    ["We", "visited", "the", "museum", "last", "weekend"],
    ["He", "is", "studying", "computer", "science", "at", "university"],
    ["The", "weather", "is", "nice", "today,", "isn't", "it?"],
    ["She", "likes", "to", "cook", "delicious", "meals", "for", "her", "family"],
    ["The", "movie", "we", "watched", "last", "night", "was", "really", "interesting"],
]

# 모델 생성
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
# 저장된 모델 사용
# model = Word2Vec.load("word2vec.model")

word = "guitar"

# 단어의 벡터값 확인
vector = model.wv[word]
print(f"'{word}' 벡터값{vector.shape}: {vector}")
'''
'guitar' 벡터값(100,): [-4.9783168e-03 -1.2840523e-03  3.2820590e-03 -6.4153541e-03
 -9.7030792e-03 -9.2653455e-03  9.0174852e-03  5.3712665e-03
......
  2.5708610e-04  3.6965334e-04  3.9439322e-03 -9.4637852e-03
  9.7037787e-03 -6.9748540e-03  5.7620732e-03 -9.4311778e-03]
'''

# 특정 단어와 유사한 단어 찾기
most_similar_words = model.wv.most_similar(word, topn=5)
for similar_word, score in most_similar_words:
    print(f"'{word}', '{similar_word}' 유사도: {score:.2f}")
'''
'guitar', 'cook' 유사도: 0.19
'guitar', 'computer' 유사도: 0.16
'guitar', 'likes' 유사도: 0.16
'guitar', 'meeting' 유사도: 0.13
'guitar', 'We' 유사도: 0.11
'''

# 두 단어 간의 유사도 계산
word1 = "guitar"
word2 = "books"
similarity = model.wv.similarity(word1, word2)
print(f"'{word1}', '{word2}' 유사도: {similarity:.2f}")
# 'guitar', 'books' 유사도: 0.09
