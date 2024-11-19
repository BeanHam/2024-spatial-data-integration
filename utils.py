import os
import json
import torch
import wandb
import evaluate
import argparse
import numpy as np
import bitsandbytes as bnb

from tqdm import tqdm
from trl import SFTTrainer
from datasets import load_dataset
from typing import Mapping, Iterable
from os import path, makedirs, getenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama3': '<|eot_id|>',
    'qwen': '<|im_end|>'
}

MODEL_PATHS = {
    'mistral': 'beanham/spatial_join_mistral',
    'llama3': 'beanham/spatial_join_llama3',
    #'qwen':,
    'gpt-4o-mini-2024-07-18': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AUloL5Yu'
}

QUANZATION_MAP = {
    '4bit': BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    '8bit': BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
        torch_dtype=torch.bfloat16,
    ),
}

DEFAULT_TRAINING_ARGS = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        use_mps_device=False,
        log_level='info',
        logging_first_step=True,
        evaluation_strategy='steps',
        eval_steps=25
    )

system_message = """
You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) information in GeoJson format. Please help me identify whether the sidewalk is alongside the paired road, such that the sidewalk is adjacent and parellele to the road. If it is, please return 1; otherwise, return 0.
"""

system_message_eval = """
You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) information in GeoJson format. Please help me identify whether the sidewalk is alongside the paired road, such that the sidewalk is adjacent and parellele to the road. If it is, please return 1; otherwise, return 0.
    
Please just return 0 or 1. No explaination needed.
"""

oneshot_example = """
Here are two examples:

#### EXAMPLE 1:

Sidewalk:
{'coordinates': [[-122.1212457, 47.6053648], [-122.1212721, 47.6053883]], 'type': 'LineString'}
Road:
{'coordinates': [[-122.1212277, 47.6053289], [-122.1211996, 47.60534], [-122.1210113, 47.6054163]], 'type': 'LineString'}
Label: 0

#### EXAMPLE 2:

Sidewalk:
{'coordinates': [[-122.11042249999998, 47.565170399999985], [-122.1102543, 47.5650583]], 'type': 'LineString'}
Road:
{'coordinates': [[-122.1110782, 47.564866], [-122.1109176, 47.5648958], [-122.11079, 47.5649255], [-122.1104464, 47.5650841], [-122.1103691, 47.5651116]], 'type': 'LineString'}
Label: 1

#### 
Please help me analyze the follow pair: 

"""


def get_dataset_slices(dataset: str) -> dict:
    """
    Returns a dictionary of subsets of the training, validation, and test splits of a dataset.
    """

    # Download the dataset splits, including the dataset version if specified
    train = load_dataset(dataset, split='train')
    val = load_dataset(dataset, split='val')
    test = load_dataset(dataset, split='test')

    # Return the dictionary of dataset splits
    return {
      'train': train,
      'val': val,
      'test': test
      }

def get_model_and_tokenizer(model_id: str, 
                            quantization_type: str='', 
                            gradient_checkpointing: bool=True, 
                            device: str='auto') -> tuple[AutoModel, AutoTokenizer]:
    """
    Returns a Transformers model and tokenizer for fine-tuning. If quantization_type is provided, the model will be quantized and prepared for training.
    """

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     quantization_config=QUANZATION_MAP[quantization_type], 
                                                     device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     device_map=device)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare the model for training if quantization is requested
    if quantization_type:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def find_lora_modules(model: AutoModel, 
                      include_modules: Iterable=(bnb.nn.Linear4bit), 
                      exclude_names: Iterable=('lm_head')) -> list[str]:
    """
    Returns a list of the modules to be tuned using LoRA.
    """

    # Create a set to store the names of the modules to be tuned
    lora_module_names = set()

    # Iterate over the model and find the modules to be tuned
    for name, module in model.named_modules():

        # Check if the module is in the list of modules to be tuned
        if any(isinstance(module, include_module) for include_module in include_modules):

            # Split the name of the module and add it to the set
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Return the list of module names to be tuned, excluding any names in the exclude list
    return [name for name in list(lora_module_names) if name not in exclude_names]

def get_lora_model(model: AutoModel,
                   matrix_rank: int=8,
                   scaling_factor: int=32,
                   dropout: float=0.05,
                   bias: str='none',
                   task_type: str='CAUSAL_LM',
                   include_modules: Iterable=(bnb.nn.Linear4bit),
                   exclude_names: Iterable=('lm_head')) -> AutoModel:
    """
    Returns a model with LoRA applied to the specified modules.
    """

    config = LoraConfig(
        r=matrix_rank,
        lora_alpha=scaling_factor,
        target_modules=find_lora_modules(model, include_modules, exclude_names),
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )

    return get_peft_model(model, config)

def format_data_as_instructions(data: Mapping, 
                                tokenizer: AutoTokenizer) -> list[str]:
    """
    Formats text data as instructions for the model. Can be used as a formatting function for the trainer class.
    """

    output_texts = []

    # Iterate over the data and format the text
    for i in tqdm(range(len(data['sidewalk'])), desc='Formatting data'):
        sidewalk = "\nSidewalk:\n"+str(data['sidewalk'][i])
        road = "\n\nRoad:\n"+str(data['road'][i])
        chat = [
          {"role": "user", "content": system_message+sidewalk+road},
          {"role": "assistant", "content": "\nLabel: "+str(data['label'][i])},      
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)

    return output_texts

    
def get_default_trainer(model: AutoModel,
                tokenizer: AutoTokenizer,
                train_dataset: Mapping,
                eval_dataset: Mapping=None,
                formatting_func: callable=format_data_as_instructions,                
                max_seq_length: int=974,
                training_args: TrainingArguments=None) -> SFTTrainer:
    """
    Returns the default trainer for fine-tuning a summarization model based on the specified training config.
    """

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args if training_args else DEFAULT_TRAINING_ARGS,
        formatting_func=formatting_func,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=max_seq_length,
        packing=False
    )

    return trainer

def evaluate_model(model: AutoModelForCausalLM, 
                   tokenizer: AutoTokenizer, 
                   data: Iterable,
                   max_tokens: int=2048,
                   min_new_tokens: int=1,
                   max_new_tokens: int=10,
                   remove_suffix: str=None,
                   fewshot: str=None) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []
    if fewshot=='True': 
        message=system_message_eval+oneshot_example
    else: 
        message=system_message_eval

    # Iterate over the test set
    for idx in tqdm(range(len(data))):
                
        sidewalk = "\nSidewalk:\n"+str(data['sidewalk'][idx])
        road = "\n\nRoad:\n"+str(data['road'][idx])
        chat = [{"role": "user", "content": message+sidewalk+road}]
        input_data = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)    
        
        ## decoding
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=input_data, 
                                       max_tokens=max_tokens,
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        decoded = decoded.replace(remove_suffix, '')
        model_outputs.append(decoded)
               
    return model_outputs

def generate_from_prompt(model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         input_data: str,
                         max_tokens: int=2048,
                         min_new_tokens: int=1,
                         max_new_tokens: int=10) -> str:
    """
    Generate and decode output from a Transformers model using a prompt.
    """

    # Calculate the position of the start of the output string
    start_decode = len(tokenizer.encode(input_data, truncation=True, max_length=max_tokens))

    # Encode the input string
    input_ids = tokenizer(input_data, return_tensors='pt', truncation=True, max_length=max_tokens).to(model.device)

    # Generate text from prompt
    with torch.no_grad():
        output = model.generate(**input_ids, 
                                max_new_tokens=max_new_tokens, 
                                min_new_tokens=min_new_tokens, 
                                pad_token_id=tokenizer.eos_token_id,
                                temperature=0.0,
                                do_sample=False)
        
    # Decode the output string, removing the special tokens and any suffixes
    decoded = tokenizer.decode(output[0][start_decode:])

    return decoded

def evaluate_gpt(test, client, model, fewshot):
    
    #--------------
    # inference
    #--------------
    model_outputs = []
    if fewshot=='True': 
        message=system_message_eval+oneshot_example
    else: 
        message=system_message_eval
            
    for idx in tqdm(range(len(test))):
        
        sidewalk = "Sidewalk: "+str(test['sidewalk'][idx])
        road = "Road: "+str(test['road'][idx])            
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": message},
                {"role": "user", "content": sidewalk+road},
            ],
            temperature=0,
            max_tokens=10,
            top_p=1
        )
        model_outputs.append(response.choices[0].message.content)

    return model_outputs
    
