import wandb
import evaluate
import argparse

from utils import *
from os import path, makedirs
from huggingface_hub import login as hf_login
from datasets import load_dataset,concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def process_train_data(train, metric_name, metric_value):
    
    if metric_name == 'degree':
        positive = train.filter(lambda x: ((x['label']==1) & (x['min_angle']<=metric_value)))
        negative = train.filter(lambda x: ((x['label']==0) & (x['min_angle']>metric_value)))
        new_train=concatenate_datasets([positive,negative]).shuffle()
    elif metric_name == 'distance':
        positive = train.filter(lambda x: ((x['label']==1) & (x['euc_dist']>=metric_value)) )
        negative = train.filter(lambda x: ((x['label']==0) & (x['euc_dist']<metric_value)) )
        new_train=concatenate_datasets([positive,negative]).shuffle()
        
    return new_train

if __name__ == '__main__':
            
    #-------------------
    # parameters
    #-------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='bert')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--metric_name', type=str, default='degree')
    parser.add_argument('--metric_value', type=int, default=1)
    args = parser.parse_args()
    
    args.model_repo = MODEL_REPOS[args.model_id]
    args.output_dir = f"outputs_{args.model_id}/{args.metric_name}/{args.metric_value}/" 
    args.save_dir = args.output_dir+'/final_model/'
    args.project_name = "spatial-join"
    args.wandb_name = f"{args.model_id}_{args.metric_name}_{args.metric_value}"
    args.hf_name = f"spatial_join_{args.model_id}_{args.metric_name}_{args.metric_value}"         
    if not path.exists(args.output_dir):
        makedirs(args.output_dir)
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)
    hf_login()
    wandb.login()
    wandb.init(project=args.project_name, name=args.wandb_name)
    
    # ----------------------
    # Load & Prepare Data
    # ----------------------
    print('Downloading and preparing data...')    
    def formatting_prompts_func(example):
        return { "text" : "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road']) }    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=args.max_seq_length, truncation=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_repo)
    data = load_dataset(args.dataset)
    data = data.map(formatting_prompts_func)
    tokenized_data = data.map(tokenize_function, batched=True)
    train = process_train_data(tokenized_data['train'], args.metric_name, args.metric_value)
    val = tokenized_data['val']
    
    # ----------------------
    # Load Model & Trainer
    # ----------------------
    metric = evaluate.load("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_repo, 
                                                               num_labels=2, 
                                                               torch_dtype="auto")
    training_args = TrainingArguments(
        num_train_epochs = 5,
        logging_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        output_dir = args.output_dir,
        load_best_model_at_end=True,
    )    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
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
