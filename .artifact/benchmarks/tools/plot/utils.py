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
    workload.update({
        "tp_size": tp_size,
        "is_swap": is_swap,
        "is_delta": is_delta
    })
    return workload

def parse_data(input_file):
    with open(input_file, "r") as fp:
        data = [json.loads(line) for line in fp]
    metadata = data.pop(0)
    key_metadata = extract_key_metadata(metadata)
    results = []
    for id, x in enumerate(data):
        metric = x['response']['metrics'][0]
        
        e2e_latency = x['time_elapsed'] - x['relative_start_at']
        inference_latency = metric['finished_time'] - metric['first_scheduled_time']
        
        first_token_latency = metric['first_token_time'] - metric['arrival_time']
        
        if metric['gpu_loading_time'] is None and metric['cpu_loading_time'] is None:
            gpu_loading_time = 0
            cpu_loading_time = 0
        elif metric['cpu_loading_time'] is None:
            cpu_loading_time = 0
            gpu_loading_time = metric['gpu_loading_time'] - metric['arrival_time']
        else:
            gpu_loading_time = metric['gpu_loading_time'] - metric['arrival_time']
            cpu_loading_time = metric['cpu_loading_time'] - metric['arrival_time']
        
        result = {
            "id": id,
            "model": x['response']['model'],
            "time": e2e_latency,
            "type": "e2e_latency",
        }
        # results.append(result)
        
        result = {
            "id": id,
            "model": x['response']['model'],
            "time": gpu_loading_time,
            "type": "gpu_loading_time",
        }
        results.append(result)
        
        result = {
            "id": id,
            "model": x['response']['model'],
            "time": cpu_loading_time,
            "type": "cpu_loading_time",
        }
        results.append(result)
        
        result = {
            "id": id,
            "model": x['response']['model'],
            "time": inference_latency,
            "type": "inference_latency",
        }
        results.append(result)
    return key_metadata, results
    