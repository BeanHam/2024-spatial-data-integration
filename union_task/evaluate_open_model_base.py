import argparse
import numpy as np

from utils import *
from tqdm import tqdm
from os import path, makedirs
from datasets import load_dataset
from unsloth import FastLanguageModel
from huggingface_hub import login as hf_login

## evaluation function
def evaluate(model, tokenizer, data):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=2048))        
        inputs = tokenizer(text, return_tensors = "pt").to("cuda")
        response = model.generate(**inputs, max_new_tokens = 10)
        response = tokenizer.decode(response[0][start_decode:])
        outputs.append(response)
    return outputs

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llama3')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)    
    data = load_dataset(args.dataset)
    methods = ['zero_shot', 'few_shot']
    modes = ['no_exp', 'with_exp']

    #-----------------------------
    # load model
    #-----------------------------
    args.model_repo = MODEL_REPOS[args.model_id]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_repo,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True
    )
    FastLanguageModel.for_inference(model)

    #-----------------------------
    # loop through methods & modes
    #-----------------------------
    for method in methods:
        for mode in modes:
            print('=================================')
            print(f'Method: {method}...')
            print(f'Mode: {mode}...')
            
            def formatting_prompts_func(example):
                output = ""
                if method=='zero_shot':                
                    if mode=='no_exp':
                        input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
                        text = base_alpaca_prompt.format(instruction_no_exp, input, output)
                    else:
                        input = "Sidewalk: "+str(example['sidewalk'])+\
                                "\nRoad: "+str(example['road'])+\
                                "\nmin_angle: "+str(example['min_angle'])+\
                                "\nmin_distance: "+str(example['euc_dist'])
                        text = base_alpaca_prompt.format(instruction_with_exp, input, output)
                else:
                    if mode=='no_exp':
                        input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])                        
                        text = base_alpaca_prompt.format(instruction_no_exp+example_one_no_exp+example_two_no_exp, input, output)
                    else:
                        input = "Sidewalk: "+str(example['sidewalk'])+\
                                "\nRoad: "+str(example['road'])+\
                                "\nmin_angle: "+str(example['min_angle'])+\
                                "\nmin_distance: "+str(example['euc_dist'])
                        text = base_alpaca_prompt.format(instruction_with_exp+example_one_with_exp+example_two_with_exp, input, output)
                return { "text" : text}
                
            test = data['test'].map(formatting_prompts_func)
            args.save_name = f"{args.model_id}_{method}_{mode}"
            outputs=evaluate(model, tokenizer, test)
            np.save(args.save_path+args.save_name+".npy", outputs)
        
if __name__ == "__main__":
    main()
