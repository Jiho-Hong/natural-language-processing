# n-gram을 이용해 문맥에 맞는 단어 확률 구하기
# `아기/Noun + 를/Josa` 이후 `낫는다/Verb`와 `'낳는다/Verb'`가 나올 확률 구하기
# 	1. '낫다'와 '낳다'가 사용된 문장이 어절 단위로 나뉘어 저장된 파일을 열어 문장으로 만들기
# 	2. Okt를 이용해 전체 문장 토크나이징 하기
# 	3. trigram 데이터 생성하고, bigram 형태로 저장
# 	4. 문맥별 단어 빈도를 이용해 두 단어의 확률 구하기
from konlpy.tag import Okt
from nltk.util import ngrams
from nltk import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, MLEProbDist


# 원본 파일을 읽어서 문장 단위로 저장하는 함수
def get_sentence(origin_file_path):
    origin_file = open(origin_file_path, encoding='euc-kr', mode='r')

    lines = origin_file.readlines()
    sentence_list = []
    for line in lines:
        words = []
        tab_splited = line.strip().split('\t')
        for s in tab_splited:
            words.append(s.split()[0] + ' ')

        sentence_list.append(''.join(words) + '\n')

    origin_file.close()

    return '\n'.join(sentence_list)


# 낫다 원본 파일 경로
nasda_origin_file_path = './낫다.txt'
# 낫다 문장
nasda_sentence = get_sentence(origin_file_path=nasda_origin_file_path)

# 낳다 원본 파일 경로
nahda_origin_file_path = './낳다.txt'
# 낳다 문장
nahda_sentence = get_sentence(origin_file_path=nahda_origin_file_path)

# Okt 객체 생성
okt = Okt()

# 낫다 문장 + 낳다 문장
total_sentence = nasda_sentence + nahda_sentence
# 전체 문장 토크나이징
tokens = ['/'.join(e) for e in okt.pos(total_sentence)]

# trigram 생성
trigrams = list(
    ngrams(tokens, 3, pad_left=True, pad_right=True, left_pad_symbol="SOS", right_pad_symbol="SOE"))

# trigram 데이터를 bigram 형태로 사용하기 위해 저장
trigrams_as_bigrams = []
trigrams_as_bigrams.extend([((t[0], t[1]), t[2]) for t in trigrams])

# 문맥별 단어 빈도
cfd = ConditionalFreqDist(trigrams_as_bigrams)

# 조건부 확률 추정
cpd = ConditionalProbDist(cfd, MLEProbDist)

pre_word = ('아기/Noun', '를/Josa')

# '아기/Noun', '를/Josa' 이후에 '낫는다/Verb'가 나올 확률
nasda = '낫는다/Verb'
print(f'{pre_word} 이후에 ({nasda})가 나올 확률: {cpd[pre_word].prob(nasda)}')

# '아기/Noun', '를/Josa' 이후에 '낳는다/Verb'가 나올 확률
nahda = '낳는다/Verb'
print(f'{pre_word} 이후에 ({nahda})가 나올 확률: {cpd[pre_word].prob(nahda)}')
