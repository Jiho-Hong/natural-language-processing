# GPT를 이용한 문서 요약
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# 출처: Hugging Face
raw_datasets = load_dataset("xsum", split="train")

# 데이터 확인
# 'document', 'summary', 'id' 로 구성
# 204,045개의 데이터
print(f'데이터 정보\n{raw_datasets}')
print(f"첫 번째 데이터의 'document':\n{raw_datasets[0]['document']}")
print(f"첫 번째 데이터의 'summary':\n{raw_datasets[0]['summary']}")
print('\n\n')

# 100개만 사용
raw_datasets = raw_datasets.select(range(100))

# 모델 이름
MODEL_NAME = "gpt2"

# 토크나이저
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 데이터 전처리
delimiter = ' TL;DR '


def preprocess(data):
    texts = [document + delimiter + summary + tokenizer.eos_token for document, summary in
             zip(data['document'], data['summary'])]

    return tokenizer(text=texts, padding=True, truncation=True, max_length=1024, return_tensors='tf')['input_ids']


tokenized_datasets = preprocess(raw_datasets)
# 전처리 결과 확인
print(f"전처리 후 첫 번째 데이터의 input_ids 앞 10개: {tokenized_datasets[0][:10]}")
print('\n\n')

# 데이터 세트 생성
train_data, train_label = [], []
for dataset in tokenized_datasets:
    train_data.append(dataset[:-1])
    train_label.append(dataset[1:])

train_data = np.array(train_data).astype(np.int32)
train_label = np.array(train_label).astype(np.int32)

train_datasets = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_datasets = (train_datasets.shuffle(len(raw_datasets)).batch(32, drop_remainder=True))

# 모델 선언
model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

model.summary()

# 학습
model.fit(train_datasets, epochs=10)

# 요약을 위한 원본 문장 선택
idx = np.random.choice((len(raw_datasets)), 1)
text = raw_datasets[idx]['document'][0]

# 요약할 문장 인코딩
input_ids = tokenizer.encode(text + delimiter, return_tensors='tf')

# 요약 문장 생성
output = model.generate(input_ids, max_length=1024, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

print(f"원본 문장: \n{text}")
print(f"데이터 세트에 있는 요약된 문장: \n{raw_datasets[idx]['summary'][0]}")
print(f"모델에서 추출된 요약된 문장: \n{tokenizer.decode(output[0], skip_special_tokens=True)[len(text) + len(delimiter):]}")
