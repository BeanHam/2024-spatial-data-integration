# 2024-spatial-join


To finetune/evalute a model, run:

```
python finetune.py --model_id {model_id} --use_model_prompt_defaults {model_prefix} 
```

```
python evaluate.py --model_id {model_id} --use_model_prompt_defaults {model_prefix} --finetuned {True/False}
```

### Model ID {Model Prefix}:
- meta-llama/Llama-3.1-8B-Instruct {llama3}
- mistralai/Mistral-7B-Instruct-v0.3 {mistral}
- Qwen/Qwen2.5-7B-Instruct {qwen}
