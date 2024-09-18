---
language: 
  - ko
tags:
- generated_from_keras_callback
model-index:
- name: t5-base-korean-text-summary
  results: []
---

# t5-base-korean-text-summary

This model is a fine-tuning of [paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base) model using AIHUB "summary and report generation data". This model provides a short summary of long sentences in Korean.

이 모델은 paust/pko-t5-base model을 AIHUB "요약문 및 레포트 생성 데이터"를 이용하여 fine tunning 한 것입니다. 이 모델은 한글로된 장문을 짧게 요약해 줍니다.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512

text = """
주인공 강인구(하정우)는 ‘수리남에서 홍어가 많이 나는데 다 갖다버린다’는 친구 
박응수(현봉식)의 얘기를 듣고 수리남산 홍어를 한국에 수출하기 위해 수리남으로 간다. 
국립수산과학원 측은 “실제로 남대서양에 홍어가 많이 살고 아르헨티나를 비롯한 남미 국가에서 홍어가 많이 잡힌다”며 
“수리남 연안에도 홍어가 많이 서식할 것”이라고 설명했다.

그러나 관세청에 따르면 한국에 수리남산 홍어가 수입된 적은 없다. 
일각에선 “돈을 벌기 위해 수리남산 홍어를 구하러 간 설정은 개연성이 떨어진다”는 지적도 한다. 
드라마 배경이 된 2008~2010년에는 이미 국내에 아르헨티나, 칠레, 미국 등 아메리카산 홍어가 수입되고 있었기 때문이다. 
실제 조봉행 체포 작전에 협조했던 ‘협력자 K씨’도 홍어 사업이 아니라 수리남에 선박용 특수용접봉을 파는 사업을 하러 수리남에 갔었다.
"""

inputs = ["summarize: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)
```


## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: None
- training_precision: float16

### Training results



### Framework versions

- Transformers 4.22.1
- TensorFlow 2.10.0
- Datasets 2.5.1
- Tokenizers 0.12.1
