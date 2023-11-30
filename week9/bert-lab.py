# BERT를 이용한 개체명 인식
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from transformers import BertTokenizerFast, TFBertModel

# 데이터 불러오기
# 출처: ukairia777/tensorflow-nlp-tutorial
url = 'https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/ner_dataset.csv'

urllib.request.urlretrieve(url=url, filename='ner_dataset.csv')

data = pd.read_csv("ner_dataset.csv", encoding="latin1")

# 데이터 확인
# 'Sentence', 'Word', 'POS', 'Tag' 으로 구성
print('데이터 확인')
print(data.head())

# Null 값 제거
print(f'Null 값 제거 전 Null 값 포함 여부: {str(data.isnull().values.any())}')
data = data.fillna(method="ffill")
print(f'Null 값 제거 후 Null 값 포함 여부: {str(data.isnull().values.any())}')

# 소문자로 변환
data['Word'] = data['Word'].str.lower()

# 태그 인코딩
enc_tag = preprocessing.LabelEncoder()
data.loc[:, "Tag"] = enc_tag.fit_transform(data["Tag"])

# 문장 단위로 단어-태그 쌍 만들기
sentences = data.groupby("Sentence #")["Word"].apply(list).values
tags = data.groupby("Sentence #")["Tag"].apply(list).values

# 단어, 태그 확인
print(f'첫 번째 문장의 단어: {sentences[0]}')
print(f'첫 번째 문장의 태그: {tags[0]}')

# 태그 유형의 개수
num_tag_type = len(data.groupby('Tag'))
"""
태그 정보
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""

# 데이터 셔플
indices = np.arange(len(sentences))
np.random.shuffle(indices)

sentences = [sentences[i] for i in indices]
tags = [tags[i] for i in indices]

# train, validation 데이터 분리
validation_rate = 0.2
num_train_data = int(len(sentences) * (1 - validation_rate))
x_train, y_train = sentences[:num_train_data], tags[:num_train_data]
x_validation, y_validation = sentences[num_train_data:], tags[num_train_data:]

# 토크나이저 선언
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 토크나이징
max_len = 128


def tokenize(data):
    input_ids, attention_masks = [], []
    for d in data:
        encoded = tokenizer.encode_plus(text=d, add_special_tokens=True, max_length=max_len, is_split_into_words=True,
                                        return_attention_mask=True, padding='max_length', truncation=True,
                                        return_tensors='np')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.vstack(input_ids), np.vstack(attention_masks)


train_input_ids, train_attention_masks = tokenize(x_train)
validation_input_ids, validation_attention_masks = tokenize(x_validation)

# 토크나이징 결과 확인
# train_input_ids 시작과 끝에 CLS(101), SEP(102) 토큰 확인
print('토크나이징 후 문장의 입력 ID와 attention mask')
print(train_input_ids[0])
print(train_attention_masks[0])


# 태그 데이터에 패딩 추가
def get_padding_data(data):
    padding_data = []
    for d in data:
        padding_data.append(np.array(d + [0] * (max_len - len(d))))

    return np.array(padding_data)


train_tags = get_padding_data(y_train)
validation_tags = get_padding_data(y_validation)

# 모델 선언
input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')

bert_model = TFBertModel.from_pretrained(model_name)
bert_output = bert_model(input_ids, attention_mask=attention_masks, return_dict=True)
embedding = tf.keras.layers.Dropout(0.3)(bert_output['last_hidden_state'])

output = tf.keras.layers.Dense(num_tag_type, activation='softmax')(embedding)
model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=[output])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 학습
model.fit([train_input_ids, train_attention_masks], train_tags,
          validation_data=([validation_input_ids, validation_attention_masks], validation_tags), epochs=20,
          batch_size=64, verbose=True)
