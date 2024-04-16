import os
import json

def parse_annotations(annotations:str):
    """annotations are in format: key1=val1,key2=val2,...
    this function parse it into dictionary as {key1: val1, key2: val2, ...}
    """
    pairs = annotations.split(",")
    parsed = {}
    for pair in pairs:
        key, val = pair.split("=")
        parsed[key] = val
    return parsed

def extract_key_metadata(metadata):
    workload = parse_annotations(metadata['workload'].split("/")[-1].removesuffix(".jsonl"))
    tp_size = metadata['sys_info']['tensor_parallel_size']
    is_swap = len(metadata['sys_info']['swap_modules']) > 0
    is_delta = len(metadata['sys_info']['delta_modules']) > 0
    is_unoptimized_delta = False
    if is_delta:
        if "unopt" in metadata['sys_info']['delta_modules'][0]['local_path']:
            is_unoptimized_delta = True
    
    workload.update({
        "tp_size": tp_size,
        "is_swap": is_swap,
        "is_delta": is_delta,
        "is_unoptimized_delta": is_unoptimized_delta,
    })
    return workload

def parse_delta_compute(data):
    results = []
    for id, x in enumerate(data):
        metric = x['response']['metrics'][0]
        
        e2e_latency = x['time_elapsed']
        
        inference_latency = metric['finished_time'] - metric['first_scheduled_time']
        
        first_token_latency = metric['first_token_time'] - metric['first_scheduled_time']
        queuing_time = metric['first_scheduled_time'] - metric['arrival_time']
        gpu_loading_time = metric['gpu_loading_time'] - metric['cpu_loading_time']
        cpu_loading_time = metric['cpu_loading_time'] - metric['first_scheduled_time']
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": e2e_latency,
            "type": "E2E Latency",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": gpu_loading_time,
            "type": "CPU -> GPU",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": cpu_loading_time,
            "type": "Disk -> CPU",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": inference_latency,
            "type": "Inference Latency",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": queuing_time,
            "type": "Queueing Delay",
        })
        
    return results

def parse_swap(data):
    results = []
    for id, x in enumerate(data):
        metric = x['response']['metrics'][0]
        
        e2e_latency = x['time_elapsed']

        inference_latency = metric['finished_time'] - metric['first_scheduled_time']
        
        first_token_latency = metric['first_token_time'] - metric['first_scheduled_time']
         
        if metric['cpu_loading_time'] is None:
            cpu_loading_time = 0
        if metric['gpu_loading_time'] is None:
            gpu_loading_time = 0
        else:
            gpu_loading_time = metric['gpu_loading_time'] - metric['arrival_time']
        queuing_time = e2e_latency - inference_latency - gpu_loading_time
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": e2e_latency,
            "type": "E2E Latency",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": gpu_loading_time,
            "type": "CPU -> GPU",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": cpu_loading_time,
            "type": "Disk -> CPU",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": inference_latency,
            "type": "Inference Latency",
        })
        results.append({
            "id": id,
            "model": x['response']['model'],
            "time": queuing_time,
            "type": "Queueing Delay",
        })
        
    return results

def parse_data(input_file):
    with open(input_file, "r") as fp:
        data = [json.loads(line) for line in fp]
    metadata = data.pop(0)
    key_metadata = extract_key_metadata(metadata)
    if key_metadata['is_delta']:
        results = parse_delta_compute(data)
    elif key_metadata['is_swap']:
        results = parse_swap(data)
    return key_metadata, results

def get_title(key_metadata):
    if key_metadata['is_swap']:
        return f"Naive vLLM, {key_metadata['distribution']}"
    if key_metadata['is_delta'] and key_metadata['is_unoptimized_delta']:
        return f"vLLM + Delta, {key_metadata['distribution']}"
    if key_metadata['is_delta'] and not key_metadata['is_unoptimized_delta']:
        return f"vLLM + Delta + Optimized I/O, {key_metadata['distribution']}"