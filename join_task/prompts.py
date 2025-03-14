## =================
## base evaluation
## =================
base_alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response: 
{}"""

## ===============================
## base evaluation -- instructions
## ===============================


## -------------------
## no heuristics
## -------------------
instruction_no_heur="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road. If it does, return 1. Otherwise, return 0. No explanation is needed."""

instruction_no_heur_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road. If it does, return 1. Otherwise, return 0. Please solve the task step by step."""

## ------------------------
## with heuristic hint
## ------------------------

## angle
instruction_heur_hint_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed. """

## distance
instruction_heur_hint_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed. """

## area
instruction_heur_hint_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed. """


## combination (angle, distance)
instruction_heur_hint_angle_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed. """


## combination (angle, area)
instruction_heur_hint_angle_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed. """


## combination (distance, area)
instruction_heur_hint_distance_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed. """


## all
instruction_heur_hint_all="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap.

If all conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed. """



## ---------------------------
## with heuristic value
## ---------------------------

## angle
instruction_heur_value_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed."""

## angle
instruction_heur_value_angle_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.

If the condition is satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistic for the condition requirement."""


## distance
instruction_heur_value_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed."""

## distance
instruction_heur_value_distance_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If the condition is satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistic for the condition requirement."""


## area
instruction_heur_value_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed."""

## area
instruction_heur_value_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If the condition is satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistic for the condition requirement."""


## combination (angle, distance)
instruction_heur_value_angle_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed."""


## combination (angle, distance)
instruction_heur_value_angle_distance_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If both conditions are satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistics for the condition requirements."""


## combination (angle, area)
instruction_heur_value_angle_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed."""


## combination (angle, area)
instruction_heur_value_angle_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If both conditions are satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistics for the condition requirements."""


## combination (distance, area)
instruction_heur_value_distance_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_distance: The minimum distance (in meters) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed."""


## combination (distance, area)
instruction_heur_value_distance_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_distance: The minimum distance (in meters) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If both conditions are satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistics for the condition requirements."""


## all
instruction_heur_value_all="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with three key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- min_distance: The minimum distance (in meters) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If all conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed."""

instruction_heur_value_all_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with three key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- min_distance: The minimum distance (in meters) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.
- Overlap: The sidewalk and road must not directly overlap, but a 10-meter buffer around each should have a certain amount of overlap. The max_area values help quantify this overlap and should not be near zero or too small.

If all conditions are satisfied, return 1. Otherwise, return 0. 

Please solve the task by analyzing the statistics for the condition requirements."""

## ===============================
## base evaluation -- examples
## ===============================

## -------------------
## no heuristics
## -------------------
examples_no_heur="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
Response: 1"""

## -------------------
## with heuristic hint
## -------------------
examples_heur_hint="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
Response: 1"""


## -------------------
## with heuristic value
## -------------------

## negative example: train[0]
## positive example: train[1]

## angle
examples_heur_value_angle="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
Response: 1"""

## distance
examples_heur_value_distance="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_distance: 0.6112785794641761
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_distance: 8.72605420848234
Response: 1"""

## area
examples_heur_value_area="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
max_area: 0.4346838047603181
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
max_area: 0.4654527079273675
Response: 1"""

## combination (angle, distance)
examples_heur_value_angle_distance="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
min_distance: 0.6112785794641761
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
min_distance: 8.72605420848234
Response: 1"""

## combination (angle, area)
examples_heur_value_angle_area="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
max_area: 0.4346838047603181
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
max_area: 0.4654527079273675
Response: 1"""


## combination (distance, area)
examples_heur_value_distance_area="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_distance: 0.6112785794641761
max_area: 0.4346838047603181
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_distance: 8.72605420848234
max_area: 0.4654527079273675
Response: 1"""

## all
examples_heur_value_all="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
min_distance: 0.6112785794641761
max_area: 0.4346838047603181
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
min_distance: 8.72605420848234
max_area: 0.4654527079273675
Response: 1"""


INSTRUCTIONS = {
    'zero_shot_no_heur': instruction_no_heur,
    
    'zero_shot_with_heur_hint_angle': instruction_heur_hint_angle,
    'zero_shot_with_heur_hint_distance': instruction_heur_hint_distance,
    'zero_shot_with_heur_hint_area': instruction_heur_hint_area,
    'zero_shot_with_heur_hint_angle_distance': instruction_heur_hint_angle_distance,
    'zero_shot_with_heur_hint_angle_area': instruction_heur_hint_angle_area,
    'zero_shot_with_heur_hint_distance_area': instruction_heur_hint_distance_area,    
    'zero_shot_with_heur_hint_all': instruction_heur_hint_all,
    
    'zero_shot_with_heur_value_angle': instruction_heur_value_angle,
    'zero_shot_with_heur_value_distance': instruction_heur_value_distance,
    'zero_shot_with_heur_value_area': instruction_heur_value_area,    
    'zero_shot_with_heur_value_angle_distance': instruction_heur_value_angle_distance,
    'zero_shot_with_heur_value_angle_area': instruction_heur_value_angle_area,
    'zero_shot_with_heur_value_distance_area': instruction_heur_value_distance_area,     
    'zero_shot_with_heur_value_all': instruction_heur_value_all,
    
    'few_shot_no_heur': instruction_no_heur + examples_no_heur,
    
    'few_shot_with_heur_hint_angle': instruction_heur_hint_angle + examples_heur_hint,
    'few_shot_with_heur_hint_distance': instruction_heur_hint_distance + examples_heur_hint,
    'few_shot_with_heur_hint_area': instruction_heur_hint_area + examples_heur_hint,
    'few_shot_with_heur_hint_angle_distance': instruction_heur_hint_angle_distance + examples_heur_hint,
    'few_shot_with_heur_hint_angle_area': instruction_heur_hint_angle_area + examples_heur_hint,
    'few_shot_with_heur_hint_distance_area': instruction_heur_hint_distance_area + examples_heur_hint,
    'few_shot_with_heur_hint_all': instruction_heur_hint_all + examples_heur_hint,
    
    'few_shot_with_heur_value_angle': instruction_heur_value_angle + examples_heur_value_angle,
    'few_shot_with_heur_value_distance': instruction_heur_value_distance + examples_heur_value_distance,
    'few_shot_with_heur_value_area': instruction_heur_value_area + examples_heur_value_area,    
    'few_shot_with_heur_value_angle_distance': instruction_heur_value_angle_distance + examples_heur_value_angle_distance,
    'few_shot_with_heur_value_angle_area': instruction_heur_value_angle_area + examples_heur_value_angle_area,
    'few_shot_with_heur_value_distance_area': instruction_heur_value_distance_area + examples_heur_value_distance_area,    
    'few_shot_with_heur_value_all': instruction_heur_value_all + examples_heur_value_all,
}

COT_INSTRUCTIONS = {
    'zero_shot_with_heur_value_angle_cot': instruction_heur_value_angle_cot,
    'zero_shot_with_heur_value_distance_cot': instruction_heur_value_distance_cot,
    'zero_shot_with_heur_value_area_cot': instruction_heur_value_area_cot,    
    'zero_shot_with_heur_value_angle_distance_cot': instruction_heur_value_angle_distance_cot,
    'zero_shot_with_heur_value_angle_area_cot': instruction_heur_value_angle_area_cot,
    'zero_shot_with_heur_value_distance_area_cot': instruction_heur_value_distance_area_cot,     
    'zero_shot_with_heur_value_all_cot': instruction_heur_value_all_cot,
    'few_shot_with_heur_value_angle_cot': instruction_heur_value_angle_cot + examples_heur_value_angle,
    'few_shot_with_heur_value_distance_cot': instruction_heur_value_distance_cot + examples_heur_value_distance,
    'few_shot_with_heur_value_area_cot': instruction_heur_value_area_cot + examples_heur_value_area,    
    'few_shot_with_heur_value_angle_distance_cot': instruction_heur_value_angle_distance_cot + examples_heur_value_angle_distance,
    'few_shot_with_heur_value_angle_area_cot': instruction_heur_value_angle_area_cot + examples_heur_value_angle_area,
    'few_shot_with_heur_value_distance_area_cot': instruction_heur_value_distance_area_cot + examples_heur_value_distance_area,    
    'few_shot_with_heur_value_all_cot': instruction_heur_value_all_cot + examples_heur_value_all,
}

## ============================
## self correction prompts
## ============================

base_alpaca_prompt_review = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
The above response is generated using heuristics. Please carefully review and identify if there are any problems with the above response."""

base_alpaca_prompt_improve = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
The above response is generated using heuristics. Please carefully review and identify if there are any problems with the above response.

{}

### Improve:
Based on your review, please give your final response, either 0 or 1. 

### Response: """



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
