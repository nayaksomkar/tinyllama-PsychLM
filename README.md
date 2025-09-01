---
base_model: unsloth/tinyllama-bnb-4bit
library_name: peft
license: apache-2.0
datasets:
- jkhedri/psychology-dataset
tags:
- unsloth
- made-with-unsloth
---

# TinyLlama Psychology Q&A Model

A fine-tuned TinyLlama 1.1B model specialized for psychology-related questions and responses.  
<a href="https://unsloth.ai">
  <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made%20with%20unsloth.png" alt="Made with Unsloth" width="150"/>
</a>

## Model Details

### Model Description

This model is a fine-tuned version of TinyLlama 1.1B specifically trained on psychology Q&A data to provide helpful psychological guidance and responses.

- **Developed by:** [Your Name/Username]  
- **Model type:** Causal Language Model (Psychology Fine-tuned)  
- **Language(s):** English  
- **License:** Apache 2.0  
- **Finetuned from model:** unsloth/tinyllama-bnb-4bit  
- **Made with:** [Unsloth](https://unsloth.ai)  

## Uses

### Direct Use
This model can be used for:
- Psychology-related question answering  
- Mental health guidance conversations  
- Educational psychology content  
- Therapeutic conversation assistance  

### Out-of-Scope Use
This model should NOT be used for:
- Professional therapy or clinical diagnosis  
- Crisis intervention  
- Replacing licensed mental health professionals  
- Medical advice or treatment recommendations  

## Training Details

### Training Data
Fine-tuned on the jkhedri/psychology-dataset containing psychology Q&A pairs.

### Training Procedure
- **Training Framework:** Unsloth + LoRA  
- **Training Platform:** Kaggle Notebooks  
- **Training Time:** 30 minutes  
- **Training Regime:** fp16 mixed precision  

#### Training Hyperparameters
- **Base Model:** TinyLlama 1.1B  
- **LoRA Rank:** 8  
- **Learning Rate:** 2e-4  
- **Batch Size:** 2 per device  
- **Gradient Accumulation:** 4 steps  
- **Epochs:** 1  
- **Max Sequence Length:** 1024  
- **Optimizer:** adamw_8bit  

## Technical Specifications

### Compute Infrastructure

#### Hardware
- **GPU:** 2x NVIDIA Tesla T4  
- **Platform:** Kaggle Notebooks  
- **Memory:** 4-bit quantization  

#### Software
- **Framework:** Unsloth  
- **Libraries:** PyTorch, Transformers, PEFT  

## How to Use

```python
from unsloth import FastLanguageModel

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained("your-username/tinyllama-psychology-lora")

# Enable fast inference
FastLanguageModel.for_inference(model)

# Example usage
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful psychologist. Answer this question:

### Input:
How can I manage anxiety?

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
