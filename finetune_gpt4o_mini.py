import argparse
from utils import *
from os import path
from openai import OpenAI

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt-4o-mini-2024-07-18', help='The model ID to fine-tune.')
    parser.add_argument('--key', type=str, help='API key to finetune GPT-4o')
    parser.add_argument('--formatted_data_dir', type=str, default='formatted_data')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--metric_name', type=str, default='degree')
    parser.add_argument('--metric_value', type=int, default=1)    
    args = parser.parse_args()
    
    # ----------------------------------
    # Set the OpenAI API key and create a client
    # ----------------------------------        
    client = OpenAI(api_key=args.key)

    # Create the training dataset
    train_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'train_{args.metric_name}_{args.metric_value}.jsonl'), "rb"),
        purpose="fine-tune"
    )
    val_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'val_{args.metric_name}_{args.metric_value}.jsonl'), "rb"),
        purpose="fine-tune"
    )
    
    # Create the fine-tuning job
    job_response = client.fine_tuning.jobs.create(
        training_file=train_response.id,
        validation_file=val_response.id,
        model=args.model_id,
        hyperparameters={"n_epochs": args.n_epochs}
    )
    
    print('Wait for the fine-tuning job to complete...')
    print('You may close the terminal now...')