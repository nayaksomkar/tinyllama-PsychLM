---
base_model: unsloth/tinyllama-bnb-4bit
library_name: peft
license: apache-2.0
datasets:
- jkhedri/psychology-dataset
tags:
- unsloth
- psychology
- lora
---

# TinyLlama Psychology Q&A Model

A fine-tuned TinyLlama 1.1B model for psychology-related Q&A, trained with [Unsloth](https://unsloth.ai).  
<a href="https://unsloth.ai">
  <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made%20with%20unsloth.png" alt="Made with Unsloth" width="150"/>
</a>

## Model Details
- **Author:** [nayaksomkar]  
- **Type:** Causal LM (Psychology fine-tuned)  
- **Language:** English  
- **Base:** unsloth/tinyllama-bnb-4bit  
- **License:** Apache 2.0  

## Intended Use
✅ Psychology Q&A  
✅ Educational content  
✅ Guidance conversations  

🚫 Not for clinical diagnosis, therapy, or crisis intervention.  

## Training
- **Data:** jkhedri/psychology-dataset  
- **Method:** LoRA fine-tuning with Unsloth  
- **Hardware:** 2× T4 GPUs (Kaggle)  
- **Duration:** ~30 mins  

### Key Hyperparams
- LoRA rank: 8  
- LR: 2e-4  
- Batch: 2 (grad acc 4)  
- Epochs: 1  
- Max length: 1024  
- Optimizer: adamw_8bit  

## How to Use
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("nayaksomkar/tinyllama-psychology-lora")
FastLanguageModel.for_inference(model)

prompt = """You are a helpful psychologist. 
Question: How can I manage anxiety?"""

inputs = tokenizer([prompt], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
