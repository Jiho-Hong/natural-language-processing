# n-gram을 이용한 문장 생성
from konlpy.tag import Okt
from konlpy.corpus import kolaw
from nltk.util import ngrams
from nltk import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, MLEProbDist
import random

# 한국 법률 말뭉치 불러오기
kolaw_text = kolaw.open('constitution.txt').read()
print(kolaw_text[:300])
'''
대한민국헌법

유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을
'''

# Okt 객체 생성
okt = Okt()

# 한국 법률 말뭉치 토크나이징
tokens = ['/'.join(e) for e in okt.pos(kolaw_text)]
print(tokens[:10])
# ['대한민국/Noun', '헌법/Noun', '\n\n/Foreign', '유구/Noun', '한/Josa', '역사/Noun', '와/Josa', '전통/Noun', '에/Josa', '빛나는/Verb']

# 토크나이징된 데이터를 bigram 단위로 생성
# 문장 앞뒤로 SOS(Start Of Sentence), EOS(End of Sentence)를 각각 추가
bigram = list(ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SOS", right_pad_symbol="SOE"))
print(bigram[:10])
# [('SOS', '대한민국/Noun'), ('대한민국/Noun', '헌법/Noun'), ('헌법/Noun', '\n\n/Foreign'), ('\n\n/Foreign', '유구/Noun'), ('유구/Noun', '한/Josa'), ('한/Josa', '역사/Noun'), ('역사/Noun', '와/Josa'), ('와/Josa', '전통/Noun'), ('전통/Noun', '에/Josa'), ('에/Josa', '빛나는/Verb')]

# 문맥별 단어 빈도 확인
cfd = ConditionalFreqDist(bigram)
word = '대한민국/Noun'
print(cfd[word].most_common())
# [('의/Josa', 5), ('은/Josa', 3), ('헌법/Noun', 1), ('임시정부/Noun', 1), ('영역/Noun', 1)]

# 조건부 확률 추정
cpd = ConditionalProbDist(cfd, MLEProbDist)
word1 = '대한민국/Noun'
word2 = '의/Josa'
print(cpd[word1].prob(word2))
# [('의/Josa', 5), ('은/Josa', 3), ('헌법/Noun', 1), ('임시정부/Noun', 1), ('영역/Noun', 1)]

# 문장 생성
# 반복 실행 시에도 동일한 결과 출력을 위해 seed 고정
random.seed(a=42)
# 시작 단어
current_word = "SOS"
# 생성된 단어들을 저장할 리스트
generated_words = []
while True:
    # 현재 단어의 빈도를 알 수 없으면 종료
    if current_word not in cpd:
        break

    # 현재 단어를 이용해 다음 단어 생성
    next_word = cpd[current_word].generate()

    # 다음 단어가 마지막 단어로 예측되면 종료
    if next_word == "SOE":
        break

    # 형태소와 품사로 분리
    morph, pos = next_word.split("/")

    if current_word == "SOS":  # 시작 단어이면, 띄어쓰기 없이 추가
        generated_words.append(morph)
    elif pos in ['Eomi', 'Exclamation', 'Josa', 'KoreanParticle', 'Punctuation',
                 'Suffix']:  # 단어와 바로 붙어 사용되는 품사들의 형태소는 띄어쓰기 없이 추가
        generated_words.append(morph)
    else:  # 공백 + 형태소 추가
        generated_words.append(" " + morph)
    current_word = next_word

# 리스트 형태의 단어들을 문자열로 만들어 반환
print(''.join(generated_words))
'''
대한민국 헌법에 의하여 영장을 매년 1회 집회· 출판은 국민은 국회는 분리 된다. 

        제 84조 대통령이 달성 될 때에는 60일 이내에 한하여 최소한으로 구성한 5. 다만, 대통령이 정 한다. 
 ② 국교는 재해를 가진다. 
...... 
'''
