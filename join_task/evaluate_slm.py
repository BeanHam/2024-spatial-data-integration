import gc
import torch
import argparse
import numpy as np

from utils import *
from tqdm import tqdm
from os import path, makedirs
from datasets import load_dataset
from transformers import pipeline

## formating function
def formatting_prompts_func(example):
    return { "text" : "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])}
        
## evaluation function
def evaluate(classifier, tokenizer_kwargs, data):
    outputs=[]
    for text in tqdm(data['text']):
        response=classifier(text, **tokenizer_kwargs)
        outputs.append(int(response[0]['label'].split('_')[1]))
    return outputs

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='bert')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--metric_name', type=str, default='degree')
    args = parser.parse_args(args=[])
    
    args.model_repo = MODEL_REPOS[args.model_id]
    args.save_path=f'inference_results/{args.metric_name}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
    if args.metric_name == 'degree':
            args.metric_values = [1,2,5,10,20]
    elif args.metric_name == 'distance':
        args.metric_values = [1,2,3,4,5]

    #-------------------
    # load dataset
    #-------------------
    data = load_dataset(args.dataset)
    test = data['test'].map(formatting_prompts_func)
    
    #---------------------------
    # loop through metric values
    #---------------------------
    for metric_value in args.metric_values:
        print('=====================================================')
        print(f'{args.metric_name}: {metric_value}...')        
        print('   -- Getting model and tokenizer...')
        args.model_path = MODEL_PATHS[f"{args.model_id}_{args.metric_name}_{metric_value}"]
        args.save_name = f"{args.model_id}_{args.metric_name}_{metric_value}"
        classifier = pipeline("text-classification", model=args.model_path, tokenizer=args.model_repo)
        tokenizer_kwargs = {'padding':'max_length','truncation':True,'max_length':512}
        outputs=evaluate(classifier, tokenizer_kwargs, test)
        np.save(args.save_path+args.save_name+".npy", outputs)
        
        ## clear memory for next metric value
        classifier.model.cpu()
        del classifier
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()
