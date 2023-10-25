# KoNLPy로 품사 태깅하기
from konlpy.tag import Okt
from konlpy.corpus import kolaw
from collections import Counter

# 텍스트
text = '고려의 통치 시스템은 성종 대에 이르러 정착되었습니다.'

# Okt 객체 생성
okt = Okt()

# 텍스트를 형태소 단위로 나눔
print(okt.morphs(text))
# ['고려', '의', '통치', '시스템', '은', '성종', '대에', '이르러', '정착', '되었습니다', '.']

# 텍스트에서 명사 추출
print(okt.nouns(text))
# ['고려', '통치', '시스템', '성종', '정착']

# 텍스트에서 구 추출
# ['고려', '고려의 통치', '고려의 통치 시스템', '성종', '정착', '통치', '시스템']

# 텍스트를 형태소 단위로 나누고, 각 형태소의 품사 정보를 태깅
print(okt.pos(text))
# [('고려', 'Noun'), ('의', 'Josa'), ('통치', 'Noun'), ('시스템', 'Noun'), ('은', 'Josa'), ('성종', 'Noun'), ('대에', 'Verb'), ('이르러', 'Verb'), ('정착', 'Noun'), ('되었습니다', 'Verb'), ('.', 'Punctuation')]

# Okt의 품사 태그 정보
print(okt.tagset)
# {'Adjective': '형용사', 'Adverb': '부사', 'Alpha': '알파벳', 'Conjunction': '접속사', 'Determiner': '관형사', 'Eomi': '어미', 'Exclamation': '감탄사', 'Foreign': '외국어, 한자 및 기타기호', 'Hashtag': '트위터 해쉬태그', 'Josa': '조사', 'KoreanParticle': '(ex: ㅋㅋ)', 'Noun': '명사', 'Number': '숫자', 'PreEomi': '선어말어미', 'Punctuation': '구두점', 'ScreenName': '트위터 아이디', 'Suffix': '접미사', 'Unknown': '미등록어', 'Verb': '동사'}

# 한국 법률 말뭉치 불러오기
kolaw_text = kolaw.open('constitution.txt').read()
print(kolaw_text[:300])
'''
대한민국헌법

유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을
'''

# 가장 많이 사용되는 명사 5개 찾기
kolaw_nouns = okt.nouns(kolaw_text)
words = [word for word, cnt in Counter(kolaw_nouns).most_common()]
print(words[:5])
# ['제', '법률', '정', '수', '대통령']
