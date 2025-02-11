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
    'llama3_degree_50': 'beanham/spatial_union_llama3_degree_50',
    'llama3_degree_60': 'beanham/spatial_union_llama3_degree_60',
    'llama3_degree_70': 'beanham/spatial_union_llama3_degree_70',
    'llama3_degree_80': 'beanham/spatial_union_llama3_degree_80',
    'llama3_degree_90': 'beanham/spatial_union_llama3_degree_90',
    'mistral_degree_1': 'beanham/spatial_join_mistral_degree_1',
    'mistral_degree_2': 'beanham/spatial_join_mistral_degree_2',
    'mistral_degree_5': 'beanham/spatial_join_mistral_degree_5',
    'mistral_degree_10': 'beanham/spatial_join_mistral_degree_10',
    'mistral_degree_20': 'beanham/spatial_join_mistral_degree_20', 
    '4o_mini_degree_1':' ft:gpt-4o-mini-2024-07-18:uw-howe-lab::ApLQ6JBH',
    '4o_mini_degree_2': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::At1OE2Tj',
    '4o_mini_degree_5': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::At1CaAjK',
    '4o_mini_degree_10': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::ApoepRH8',
    '4o_mini_degree_20': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AuRjRqQg',
    'bert_degree_1': 'beanham/spatial_join_bert_degree_1',
    'bert_degree_2': 'beanham/spatial_join_bert_degree_2',
    'bert_degree_5': 'beanham/spatial_join_bert_degree_5',
    'bert_degree_10': 'beanham/spatial_join_bert_degree_10',
    'bert_degree_20': 'beanham/spatial_join_bert_degree_20'
}

MODEL_STEPS_PATHS = {
    'llama3_degree_step_1': 'outputs_llama3/degree/5/checkpoint-753',
    'llama3_degree_step_2': 'outputs_llama3/degree/5/checkpoint-1506',
    'llama3_degree_step_3': 'outputs_llama3/degree/5/checkpoint-2259',
    'llama3_degree_step_4': 'outputs_llama3/degree/5/checkpoint-3012',
    'llama3_degree_step_5': 'outputs_llama3/degree/5/checkpoint-3765',
    'mistral_degree_step_1': 'outputs_mistral/degree/10/checkpoint-754',
    'mistral_degree_step_2': 'outputs_mistral/degree/10/checkpoint-1508',
    'mistral_degree_step_3': 'outputs_mistral/degree/10/checkpoint-2262',
    'mistral_degree_step_4': 'outputs_mistral/degree/10/checkpoint-3016',
    'mistral_degree_step_5': 'outputs_mistral/degree/10/checkpoint-3770',
}

## ============================
## finetuning instructions
## ============================
instruction="You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

gpt_instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."""


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
