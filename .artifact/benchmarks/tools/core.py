import copy
import requests
import threading
import numpy as np
import sched, time
from typing import List
from timeit import default_timer as timer

s = sched.scheduler(time.monotonic, time.sleep)
threads = []
inference_results = []


def parse_annotation(annotations):
    annos = []
    annotations = annotations.split(",")
    for annotation in annotations:
        print(annotation)
        anno = annotation.split("=")
        annos.append({"name": anno[0], "value": anno[1]})
    return annos


def request_thread(
    endpoint,
    req,
    start_time,
    global_start_time,
):
    global inference_results
    res = requests.post(endpoint + "/v1/completions", json=req)
    end_time = timer()
    if res.status_code != 200:
        print(f"Failed to issue request: {res.text}")
    res = {
        "response": res.json(),
        "time_elapsed": end_time - start_time,
        "relative_start_at": start_time - global_start_time,
    }
    inference_results.append(res)
    return res


def async_issue_requests(endpoint, reqs, global_start_time):
    global threads
    for req in reqs:
        start_time = timer()
        thread = threading.Thread(
            target=request_thread,
            args=(
                endpoint,
                req,
                start_time,
                global_start_time,
            ),
        )
        threads.append(thread)
        thread.start()


def issue_queries(endpoint, queries):
    print("Issuing queries")
    time_step = 0.1
    global threads
    global inference_results
    time_range = [x["timestamp"] for x in queries]
    max_time = max(time_range) + 1
    # execute for one more second
    start = timer()
    for time in np.arange(0, max_time, time_step):
        sub_queries = [
            x
            for x in queries
            if x["timestamp"] <= time and x["timestamp"] > time - time_step
        ]
        if len(sub_queries) > 0:
            print(f"sending {len(sub_queries)} queries at {time}")
            s.enter(
                time,
                1,
                async_issue_requests,
                argument=(
                    endpoint,
                    sub_queries,
                    start,
                ),
            )

    s.run(blocking=True)
    print(f"total threads: {len(threads)}")
    [thread.join() for thread in threads]
    end = timer()
    return {"results": inference_results, "total_elapsed": end - start}


def warmup(endpoint: str, workload: List, base_model: str, warmup_strategy: str):
    print("Warming up starts")
    if warmup_strategy == "random":
        reqs = np.random.choice(workload, size=10)
    req = [x for x in reqs if x["model"] != base_model][0]
    req = copy.deepcopy(req)
    req["timestamp"] = 0
    print(req)
    res = requests.post(endpoint + "/v1/completions", json=req)
    if res.status_code != 200:
        print(f"Failed to warm up: {res.text}")
    print("Warming up ends")


def run(
    endpoints: List[str],
    workload: List,
    warmup_strategy: str,
    base_model: str,
    sysinfo: dict,
):
    global inference_results
    warmup(endpoints[0], workload, base_model, warmup_strategy)
    issue_queries(endpoints[0], workload)
    return inference_results


def get_sys_info(endpoint: str):
    return requests.get(endpoint + "/sysinfo").json()
