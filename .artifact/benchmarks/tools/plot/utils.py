import os
import json

color_palette = {
    "general": [
        "#90a0c8",
        "#f19e7b",
        "#72ba9d",
        "#bfc8c9" "#f9daad",
        "#fbe9d8",
    ]
}


def get_system_name(sys):
    sys = sys.lower()
    if "vllm" in sys:
        return "Baseline-1"
    if "deltaserve" in sys:
        if "prefetch" not in sys:
            return "Ours"
        else:
            return "Ours+"
    return "Unknown"


system_color_mapping = {"Baseline-1": "#90a0c8", "Ours": "#f19e7b", "Ours+": "#72ba9d"}


def parse_annotations(annotations: str):
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
    workload = parse_annotations(
        metadata["workload"].split("/")[-1].removesuffix(".jsonl")
    )
    gen_tokens = metadata["workload"].split("/")[-2].removeprefix("gen_")

    tp_size = metadata["sys_info"]["tensor_parallel_size"]
    is_swap = len(metadata["sys_info"]["swap_modules"]) > 0
    is_delta = len(metadata["sys_info"]["delta_modules"]) > 0
    enable_prefetch = True
    bitwidth = 4
    if "enable_prefetch" in metadata["sys_info"]:
        if not metadata["sys_info"]["enable_prefetch"]:
            enable_prefetch = False
    is_nvme = False
    if is_swap:
        bitwidth = 16
        if metadata["sys_info"]["swap_modules"][0]["local_path"].startswith("/scratch"):
            is_nvme = True
    if is_delta:
        if metadata["sys_info"]["delta_modules"][0]["local_path"].startswith(
            "/scratch"
        ):
            is_nvme = True
        if "2b" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            bitwidth = 2
        if "4b" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            bitwidth = 4
    is_unoptimized_delta = False
    if is_delta:
        if "unopt" in metadata["sys_info"]["delta_modules"][0]["local_path"]:
            is_unoptimized_delta = True

    workload.update(
        {
            "bitwidth": bitwidth,
            "tp_size": tp_size,
            "is_swap": is_swap,
            "is_delta": is_delta,
            "is_unoptimized_delta": is_unoptimized_delta,
            "gen_tokens": gen_tokens,
            "is_nvme": is_nvme,
            "enable_prefetch": enable_prefetch,
        }
    )
    return workload


def parse_delta_compute(data):
    results = []
    for id, x in enumerate(data):
        metric = x["response"]["metrics"][0]
        e2e_latency = metric["finished_time"] - metric["arrival_time"]
        first_token_latency = metric["first_token_time"] - metric["arrival_time"]
        queuing_time = metric["first_scheduled_time"] - metric["arrival_time"]
        gpu_loading_time = metric["gpu_loading_time"] - metric["cpu_loading_time"]
        cpu_loading_time = metric["cpu_loading_time"] - metric["first_scheduled_time"]
        inference_time = metric["finished_time"] - metric["gpu_loading_time"]
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": e2e_latency,
                "type": "E2E Latency",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": first_token_latency,
                "type": "TTFT",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": gpu_loading_time + cpu_loading_time,
                "type": "Loading",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": inference_time,
                "type": "Inference",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": queuing_time,
                "type": "Queueing",
            }
        )

    return results


def parse_swap(data):
    results = []
    for id, x in enumerate(data):
        metric = x["response"]["metrics"][0]
        cpu_loading_time = 0
        e2e_latency = metric["finished_time"] - metric["arrival_time"]
        first_token_latency = metric["first_token_time"] - metric["arrival_time"]
        if metric["start_loading_time"] is None:
            gpu_loading_time = 0
            queuing_time = metric["first_scheduled_time"] - metric["arrival_time"]
        else:
            gpu_loading_time = metric["gpu_loading_time"] - metric["start_loading_time"]
            queuing_time = metric["start_loading_time"] - metric["arrival_time"]
        inference_time = metric["finished_time"] - metric["first_scheduled_time"]
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": e2e_latency,
                "type": "E2E Latency",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": first_token_latency,
                "type": "TTFT",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": gpu_loading_time,
                "type": "Loading",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": inference_time,
                "type": "Inference",
            }
        )
        results.append(
            {
                "id": id,
                "model": x["response"]["model"],
                "time": queuing_time,
                "type": "Queueing",
            }
        )
    return results


def parse_data(input_file):
    with open(input_file, "r") as fp:
        data = [json.loads(line) for line in fp]
    metadata = data.pop(0)
    key_metadata = extract_key_metadata(metadata)
    if key_metadata["is_delta"]:
        results = parse_delta_compute(data)
    elif key_metadata["is_swap"]:
        results = parse_swap(data)
    return key_metadata, results


def get_title(key_metadata):
    sys = "Unknown"
    hardware = "Unknown"
    if key_metadata["is_swap"]:
        sys = "\\text{vLLM}"
    if key_metadata["is_delta"]:
        sys = "\\text{DeltaServe}"
        sys += f"({key_metadata['bitwidth']}bit)"
        if key_metadata["is_unoptimized_delta"]:
            pass
        if key_metadata["is_delta"] and not key_metadata["is_unoptimized_delta"]:
            sys += "\\text{+I/O}"
        if key_metadata["enable_prefetch"]:
            sys += "\\text{+Prefetch}"
    workload = ""
    # workload = "\\text{<>}, ".replace("<>", key_metadata["distribution"])
    workload += f"\lambda={key_metadata['ar']}"
    if key_metadata["is_nvme"]:
        hardware = "\\text{NVMe}"
    else:
        hardware = "\\text{NFS}"
    sys = "\Large{" + sys + "}"
    workload = "\Large{" + workload + "}"
    hardware = "\Large{" + hardware + "}"
    return f"${sys}, {workload}, {hardware}$"


def get_sys_name(key_metadata):
    sys = "Unknown"
    hardware = "Unknown"
    if key_metadata["is_swap"]:
        sys = "\\text{vLLM}"
    if key_metadata["is_delta"]:
        sys = "\\text{DeltaServe}"
        sys += str(key_metadata["bitwidth"]) + "\\text{bit}"
        if key_metadata["is_unoptimized_delta"]:
            pass
        if key_metadata["is_delta"] and not key_metadata["is_unoptimized_delta"]:
            sys += "\\text{+I/O}"
        if key_metadata["enable_prefetch"]:
            sys += "\\text{+Prefetch}"
    if key_metadata["is_nvme"]:
        hardware = "\\text{NVMe}"
    else:
        hardware = "\\text{NFS}"
    sys = "\Large{" + sys + "}"
    hardware = "\Large{" + hardware + "}"
    return f"${sys}, {hardware}$"
