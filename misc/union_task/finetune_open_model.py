import gc
import torch
import wandb
import argparse

from utils import *
from trl import SFTTrainer
from os import path, makedirs
from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from datasets import load_dataset,concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported


def process_train_data(train, metric_name, metric_value):
    
    if metric_name == 'degree':
        positive = train.filter(lambda x: ((x['label']==1) & (x['min_angle']<=metric_value)))
        negative = train.filter(lambda x: ((x['label']==0) & (x['min_angle']>metric_value)))
        new_train=concatenate_datasets([positive,negative]).shuffle()
    elif metric_name == 'area':
        positive = train.filter(lambda x: ((x['label']==1) & (x['max_area']>=metric_value)) )
        negative = train.filter(lambda x: ((x['label']==0) & (x['max_area']<metric_value)) )
        new_train=concatenate_datasets([positive,negative]).shuffle()
        
    return new_train

if __name__ == '__main__':
            
    #-------------------
    # parameters
    #-------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llama3')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_union_dataset')
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--metric_name', type=str, default='degree')
    args = parser.parse_args()
    
    args.model_repo = MODEL_REPOS[args.model_id]
    args.project_name = "spatial_union"
    if args.metric_name == 'degree':
        #args.metric_values = [1,2,3,4,5]
        args.metric_values = [2,3,4,5]
    elif args.metric_name == 'area':
        args.metric_values = [0.5,0.6,0.7,0.8,0.9]
    hf_login()
    wandb.login()

    #---------------------------
    # loop through metric values
    #---------------------------
    for metric_value in args.metric_values:
        print(f'{args.metric_name}: {metric_value}...')
        args.metric_value=metric_value
        args.output_dir = f"outputs_{args.model_id}/{args.metric_name}/{args.metric_value}/" 
        args.save_dir = args.output_dir+'/final_model/'
        args.wandb_name = f"unsloth_{args.model_id}_{args.metric_name}_{args.metric_value}"
        args.hf_name = f"spatial_union_{args.model_id}_{args.metric_name}_{args.metric_value}"        
        if not path.exists(args.output_dir):
            makedirs(args.output_dir)
        if not path.exists(args.save_dir):
            makedirs(args.save_dir)
        
        # HF & wandb Login    
        wandb.init(project=args.project_name, name=args.wandb_name)
            
        # ----------------------
        # Load Model & Tokenizer
        # ----------------------
        print('Getting model and tokenizer...')
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_repo,
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = True
        )
        
        print('Getting LoRA model...')
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False, 
            loftq_config = None,
        )
        
        # ----------------------
        # Load & Prepare Data
        # ----------------------
        print('Downloading and preparing data...')    
        def formatting_prompts_func(example):
            input       = "Sidewalk 1: "+str(example['sidewalk'])+"\nSidewalk 2: "+str(example['road'])
            output      = "Label: "+str(example['label'])
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            return { "text" : text, }    
        
        EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
        data = load_dataset(args.dataset)
        data = data.map(formatting_prompts_func)
        train = data['train']
        train = process_train_data(train, args.metric_name, args.metric_value)
        val = data['val']
                
        # -----------------------
        # Set Training Parameters
        # -----------------------
        print('Instantiating trainer...')
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train,
            eval_dataset = val,
            dataset_text_field = "text",
            max_seq_length = args.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                per_device_eval_batch_size = 2,
                gradient_accumulation_steps = 4,        
                warmup_ratio =0.03,        
                num_train_epochs = 5, # Set this for 1 full training run.
                max_steps = -1,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_strategy="steps",
                save_strategy="steps",
                evaluation_strategy="steps",
                logging_steps = 0.2,
                eval_steps = 0.2,
                save_steps = 0.2,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = args.output_dir,
                report_to = "wandb", # Use this for WandB etc
                load_best_model_at_end=True,
            ),
        )
        
        # -----------------------            
        # Fine-tune model
        # -----------------------    
        print('Fine-tuning model...')
        trainer_stats = trainer.train()
    
        # -----------------------    
        # Save model to hub
        # -----------------------    
        print('Saving model and tokenizer...')
        trainer.model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        trainer.model.push_to_hub(args.hf_name, use_auth_token=True)
        wandb.finish()

        ## clear memory for next metric value
        trainer.model.cpu()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
