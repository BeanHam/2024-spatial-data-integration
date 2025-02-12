MODEL_REPOS = {
    'bert': 'google-bert/bert-base-uncased',
    't5': 'google-t5/t5-large',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3': 'meta-llama/Llama-3.1-8B-Instruct',
    '4o_mini': 'gpt-4o-mini-2024-07-18',
    'o3_mini': 'o3-mini-2025-01-31',
    '4o': 'gpt-4o-2024-08-06',
}

MODEL_PATHS = {
    'llama3_degree_1': 'beanham/spatial_union_llama3_degree_1',
    'llama3_degree_2': 'beanham/spatial_union_llama3_degree_2',
    'llama3_degree_3': 'beanham/spatial_union_llama3_degree_3',
    'llama3_degree_4': 'beanham/spatial_union_llama3_degree_4',
    'llama3_degree_5': 'beanham/spatial_union_llama3_degree_5',
    'mistral_degree_1': 'beanham/spatial_union_mistral_degree_1',
    'mistral_degree_2': 'beanham/spatial_union_mistral_degree_2',
    'mistral_degree_3': 'beanham/spatial_union_mistral_degree_3',
    'mistral_degree_4': 'beanham/spatial_union_mistral_degree_4',
    'mistral_degree_5': 'beanham/spatial_union_mistral_degree_5', 
    '4o_mini_degree_1': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::B0CFV6qQ',
    '4o_mini_degree_2': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::B0D25V3z',
    '4o_mini_degree_3': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::B0CvguXY',
    '4o_mini_degree_4': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::B0D2vh6l',
    '4o_mini_degree_5': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::B0CuidcL',
    'bert_degree_1': 'beanham/spatial_union_bert_degree_1',
    'bert_degree_2': 'beanham/spatial_union_bert_degree_2',
    'bert_degree_3': 'beanham/spatial_union_bert_degree_3',
    'bert_degree_4': 'beanham/spatial_union_bert_degree_4',
    'bert_degree_5': 'beanham/spatial_union_bert_degree_5',
    'llama3_area_0.5': 'beanham/spatial_union_llama3_area_0.5',
    'llama3_area_0.6': 'beanham/spatial_union_llama3_area_0.6',
    'llama3_area_0.7': 'beanham/spatial_union_llama3_area_0.7',
    'llama3_area_0.8': 'beanham/spatial_union_llama3_area_0.8',
    'llama3_area_0.9': 'beanham/spatial_union_llama3_area_0.9',
    'mistral_area_0.5': 'beanham/spatial_union_mistral_area_0.5',
    'mistral_area_0.6': 'beanham/spatial_union_mistral_area_0.6',
    'mistral_area_0.7': 'beanham/spatial_union_mistral_area_0.7',
    'mistral_area_0.8': 'beanham/spatial_union_mistral_area_0.8',
    'mistral_area_0.9': 'beanham/spatial_union_mistral_area_0.9', 
    '4o_mini_area_0.5':' ft:gpt-4o-mini-2024-07-18:uw-howe-lab::ApLQ6JBH',
    '4o_mini_area_0.6': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::At1OE2Tj',
    '4o_mini_area_0.7': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::At1CaAjK',
    '4o_mini_area_0.8': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::ApoepRH8',
    '4o_mini_area_0.9': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AuRjRqQg',
    'bert_area_0.5': 'beanham/spatial_union_bert_area_0.5',
    'bert_area_0.6': 'beanham/spatial_union_bert_area_0.6',
    'bert_area_0.7': 'beanham/spatial_union_bert_area_0.7',
    'bert_area_0.8': 'beanham/spatial_union_bert_area_0.8',
    'bert_area_0.9': 'beanham/spatial_union_bert_area_0.9'    
}

## ============================
## finetuning instructions
## ============================
instruction="You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) GeoJSON annotations. Your task is to determine whether these two annotations represent the same sidewalk, either fully or partially, To do this, assess whether they overlap fully or partially and check if they are approximately parallel. If they represent the same sidewalk, return 1; otherwise, return 0. "

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

gpt_instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) GeoJSON annotations. Your task is to determine whether these two annotations represent the same sidewalk, either fully or partially, To do this, assess whether they overlap fully or partially and check if they are approximately parallel. If they represent the same sidewalk, return 1; otherwise, return 0."""


## ============================
## base evaluation -- zero shot
## ============================
base_alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

## ===============================
## base evaluation -- examples
## ===============================
example_one_no_exp="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
Response: {0}"""

example_one_with_exp="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
min_angle: 89.77154191724516
max_area: 0.5473559863066643
Response: {0}"""

example_two_no_exp="""

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
Response: {1}"""

example_two_with_exp="""

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
min_angle: 0.8684260514779112
max_area: 0.9832547250026938
Response: {1}"""

## ===============================
## base evaluation -- instructions
## ===============================
instruction_no_exp="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) GeoJSON. Your task is to determine whether these two annotations represent the same sidewalk, either fully or partially, by evaluating the following conditions:

1. Overlap: The two sidewalks must overlap or partially overlap. Simply connecting at the endpoints does not count as an intersection.
2. Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations.

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed. """

instruction_with_exp="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) GeoJSON, along with two key statistics:

1. min_angle: The minimum angle (in degrees) between the two sidewalks.
2. max_area: The maximum percentage of overlapping area relative to both sidewalks, considering a 10-meter buffer.

Your task is to determine whether these two annotations represent the same sidewalk, either fully or partially, by evaluating the following conditions:

1. Overlap: The two sidewalks must overlap or partially overlap. Simply connecting at the endpoints does not count as an intersection. The max_area value helps quantify this overlap.
2. Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations. The min_angle value provides a measure of this alignment.

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed."""
