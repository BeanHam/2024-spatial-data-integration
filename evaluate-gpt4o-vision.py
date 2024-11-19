import os
import json
import torch
import base64
import requests
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

prompt = """
You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) information in GeoJSON format, along with a satellite image visualizing the sidewalk (red line) and road (blue line). Please help me identify whether the sidewalk is alongside the paired road, such that the sidewalk is adjacent and parellele to the road. If it is, please return 1; otherwise, return 0.
    
Please just return 0 or 1. No explaination needed.
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def write_completion_request(prompt, base64_image, gpt_model):
    """
    Compose completion request.
    """
    
    completion = {
      "model": gpt_model,
      "messages": [
          {"role": "user",
           "content": [
               {"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
           ]}
      ],
      "max_tokens": 10
    }
    return completion

def evaluate_gpt_vision(index, data, split, paths):
    model_outputs=[]
    for i in tqdm(range(len(index))):
        img_name=index[i]
        sidewalk = "\nSidewalk:\n"+str(data['sidewalk'][i])
        road = "\n\nRoad:\n"+str(data['road'][i])
        message=prompt+sidewalk+road
        if split=='test':
            if 'positive' in img_name:
                img_path=paths['p_path']+img_name+'.png'
            else:
                img_path=paths['n_path']+img_name+'.png'
        elif split=='fp':
            img_path=paths['fp_path']+img_name+'.png'
        else:
            img_path=paths['fn_path']+img_name+'.png'
            
        base64_image = encode_image(img_path)
        completion = write_completion_request(message, base64_image, gpt_model)
        response = requests.post(api_web, headers=headers, json=completion)
        model_outputs.append(response.json()['choices'][0]['message']['content'])
    return model_outputs

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join')
    parser.add_argument('--key', type=str, default='123abc')
    args = parser.parse_args()
    args.save_path=f'inference_results/'
    paths={
        'p_path':'../2024-spatial-join-exp/join_task_imgs/positive/',
        'n_path':'../2024-spatial-join-exp/join_task_imgs/negative/',
        'fp_path':'../2024-spatial-join-exp/join_task_imgs/false_positive/',
        'fn_path':'../2024-spatial-join-exp/join_task_imgs/false_negative/'
    }
    api_web = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # ----------------------
    # Load Index & Data
    # ----------------------
    print('Downloading and preparing data...')
    with open('../2024-spatial-join-exp/join_task_data/index.txt', 'r') as f:
        index = json.load(f)
    test_index=index['test']    
    fp_index=index['fp']
    fn_index=index['fn']
    
    data=load_dataset(args.dataset)
    test_data=data['test']
    fp_data=data['fp']
    fn_data=data['fn']    

    #-----------------------
    # Inference
    #-----------------------
    test_outputs=evaluate_gpt_vision(test_index, test_data, 'test', paths)
    fp_outputs=evaluate_gpt_vision(fp_index, fp_data, 'fp', paths)
    fn_outputs=evaluate_gpt_vision(fn_index, fn_data, 'fn', paths)
    
    np.save(args.save_path+f"{args.model_id}-vision_finetuned_False_fewshot_False_test.npy", test_outputs)
    np.save(args.save_path+f"{args.model_id}-vision_finetuned_False_fewshot_False_fp.npy", fp_outputs)
    np.save(args.save_path+f"{args.model_id}-vision_finetuned_False_fewshot_False_fn.npy", fn_outputs)
    
if __name__ == "__main__":
    main()
