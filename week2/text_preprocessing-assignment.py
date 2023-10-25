# KoNLPy를 이용해 카카오톡 대화 데이터에서 가장 많이 사용되는 명사와 그 개수 찾기
from konlpy.tag import Okt
from collections import Counter

# 대화 데이터 추출 방법
# https://cs.kakao.com/helps?service=8&category=24&device=1013&locale=ko&articleId=1073188988&controllerName=help&actionName=mobileviewpage&accountLoginUrl=https%3A%2F%2Faccounts.kakao.com%2F&without_layout=false

# 카카오톡 대화 파일 열기
kakaotalk_chat_file = open('카카오톡 대화 파일(.txt) 경로', mode='r')

# 카카오톡 대화 파일 읽기
kakaotalk_chat_text = kakaotalk_chat_file.read()

# Okt 객체 생성
okt = Okt()

# 명사 추출
kakaotalk_chat_nouns = okt.nouns(kakaotalk_chat_text)

# 추출된 명사를 빈도순으로 저장
words = [(word, cnt) for word, cnt in Counter(kakaotalk_chat_nouns).most_common()]

# 출력
print(f'가장 많이 사용되는 명사는 "{words[0][0]}"이고, 그 갯수는 "{words[0][1]}"입니다.')

# 카카오톡 대화 파일 닫기
kakaotalk_chat_file.close()
