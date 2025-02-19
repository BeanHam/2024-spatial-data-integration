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
    'llama3_degree_1': 'beanham/spatial_join_llama3_degree_1',
    'llama3_degree_2': 'beanham/spatial_join_llama3_degree_2',
    'llama3_degree_5': 'beanham/spatial_join_llama3_degree_5',
    'llama3_degree_10': 'beanham/spatial_join_llama3_degree_10',
    'llama3_degree_20': 'beanham/spatial_join_llama3_degree_20',
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
    'bert_degree_20': 'beanham/spatial_join_bert_degree_20',
    'llama3_distance_1': 'beanham/spatial_join_llama3_distance_1',
    'llama3_distance_2': 'beanham/spatial_join_llama3_distance_2',
    'llama3_distance_3': 'beanham/spatial_join_llama3_distance_3',
    'llama3_distance_4': 'beanham/spatial_join_llama3_distance_4',
    'llama3_distance_5': 'beanham/spatial_join_llama3_distance_5',
    'mistral_distance_1': 'beanham/spatial_join_mistral_distance_1',
    'mistral_distance_2': 'beanham/spatial_join_mistral_distance_2',
    'mistral_distance_3': 'beanham/spatial_join_mistral_distance_3',
    'mistral_distance_4': 'beanham/spatial_join_mistral_distance_4',
    'mistral_distance_5': 'beanham/spatial_join_mistral_distance_5',
    '4o_mini_distance_1': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AsFocGzQ',
    '4o_mini_distance_2': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AsIZS7PJ',
    '4o_mini_distance_3': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AsHyOjiz',
    '4o_mini_distance_4': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AsdH1Xrx',
    '4o_mini_distance_5': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AsdOZb2b',
    'bert_distance_1': 'beanham/spatial_join_bert_distance_1',
    'bert_distance_2': 'beanham/spatial_join_bert_distance_2',
    'bert_distance_3': 'beanham/spatial_join_bert_distance_3',
    'bert_distance_4': 'beanham/spatial_join_bert_distance_4',
    'bert_distance_5': 'beanham/spatial_join_bert_distance_5',
}

MODEL_PATHS_WEAK_LABELS = {
    'llama3_degree_1_weak': 'beanham/spatial_join_llama3_degree_1_weak',
    'llama3_distance_1_weak': 'beanham/spatial_join_llama3_distance_1_weak',
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
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
Response: {0}"""

example_one_with_exp="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
min_distance: 0.6112785794641761
Response: {0}"""

example_two_no_exp="""

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
Response: {1}"""

example_two_with_exp="""

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
min_distance: 8.72605420848234
Response: {1}"""

## ===============================
## base evaluation -- instructions
## ===============================
instruction_no_exp="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) GeoJSON. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

1. Adjacency: The sidewalk and road should be in close proximity, meaning they are near each other but do not overlap or intersect.
2. Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed. """


instruction_with_exp="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) GeoJSON, along with two key statistics:

1. min_angle: The minimum angle (in degrees) between the sidewalk and the road.
2. min_distance: The minimum distance between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

1. Adjacency: The sidewalk and road should be in close proximity, meaning they are near each other but do not overlap or intersect. The min_distance value helps quantify this proximity.
2. Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed."""
