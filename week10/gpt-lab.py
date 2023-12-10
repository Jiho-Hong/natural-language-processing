# GPT를 이용한 문장 생성
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# 모델 이름
MODEL_NAME = "gpt2"

# 토크나이저
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# 모델 선언
model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)

# 입력 문장
text = 'Researchers have found a tyrannosaur’s last meal perfectly preserved inside its stomach cavity.'

# 문장 인코딩
input_ids = tokenizer.encode(text, return_tensors='tf')

# 문장 생성
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(f"모델에서 생성된 문장: \n{tokenizer.decode(output[0], skip_special_tokens=True)}")
