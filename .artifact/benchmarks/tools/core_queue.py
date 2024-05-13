import copy
import requests
import numpy as np
import time
from typing import List
from timeit import default_timer as timer
from queue import Queue
import threading

inference_results = []
ongoing_reload = False
current_model = None
threads = []


def parse_annotation(annotations):
    annos = []
    annotations = annotations.split(",")
    for annotation in annotations:
        anno = annotation.split("=")
        annos.append({"name": anno[0], "value": anno[1]})
    return annos


def request_thread(
    endpoint,
    req,
    start_time,
):
    global inference_results
    global current_model
    global threads
    if "reload" in req and req["reload"]:
        for t in threads:
            t.join()
        print(f"Sending reload request for {req['model']}", flush=True)
        req["start_loading_time"] = time.time()
        res = requests.post(
            endpoint + "/v1/reload",
            json={
                "type": "reload",
                "target": req["model"],
                "timestamp": req["timestamp"],
            },
        )
        req["finish_loading_time"] = time.time()
        if res.status_code != 200:
            print(f"Failed to issue reload request: {res.text}", flush=True)
        current_model = req["model"]

    def send_completion_requests():
        print(f"Sending completion request for {req['model']}", flush=True)
        res = requests.post(endpoint + "/v1/completions", json=req)
        print(f"response: {res.text}")
        end_time = timer()
        if res.status_code != 200:
            print(f"Failed to issue request: {res.text}", flush=True)
        res = {
            "request": req,
            "response": res.json(),
            "end_at": end_time,
            "start_at": start_time,
        }
        inference_results.append(res)

    thread = threading.Thread(target=send_completion_requests)
    while current_model != req["model"]:
        time.sleep(0.1)
    thread.start()
    threads.append(thread)
    return thread


def issue_queries(endpoint, queries):
    print("Issuing queries", flush=True)
    current_time = time.time()
    queue = Queue()
    for query in queries:
        queue.put(query)
    # not get items from the queue
    while not queue.empty():
        query = queue.get()
        # check if it is time to issue the query
        while time.time() - current_time < query["timestamp"]:
            time.sleep(0.1)
        request_thread(endpoint, query, time.time())


def warmup(
    endpoint: str,
    workload: List,
    base_model: str,
    warmup_strategy: str,
    needs_swap=True,
):
    global current_model
    print("Warming up starts", flush=True)
    if warmup_strategy == "random":
        reqs = np.random.choice(workload, size=10)
    req = [x for x in reqs if x["model"] != base_model][0]
    req = copy.deepcopy(req)
    req["timestamp"] = 0
    print(req, flush=True)
    if needs_swap:
        reload_res = requests.post(
            endpoint + "/v1/reload",
            json={
                "type": "reload",
                "target": req["model"],
                "timestamp": 0,
            },
        )
    res = requests.post(endpoint + "/v1/completions", json=req)
    if res.status_code != 200:
        print(f"Failed to warm up: {res.text}", flush=True)
    print("Warming up ends", flush=True)
    current_model = req["model"]
    return req["model"]


def prepare_queries(workloads: List, sysinfo: dict, warmup_model: str):
    swap_modules = sysinfo["swap_modules"]
    if len(swap_modules) == 0:
        pass
    else:
        # add reload workload in the middle
        current_model = warmup_model
        for idx, wk in enumerate(workloads):
            if wk["model"] != current_model:
                current_model = wk["model"]
                wk["reload"] = True
            else:
                wk["reload"] = False
    return workloads


def run(
    endpoints: List[str],
    workload: List,
    warmup_strategy: str,
    base_model: str,
    sysinfo: dict,
):
    global inference_results
    needs_swap = len(sysinfo["swap_modules"]) > 0
    selected_model = warmup(
        endpoints[0], workload, base_model, warmup_strategy, needs_swap
    )
    workload = prepare_queries(workload, sysinfo, selected_model)
    issue_queries(endpoints[0], workload)
    return inference_results


def get_sys_info(endpoint: str):
    return requests.get(endpoint + "/sysinfo").json()
