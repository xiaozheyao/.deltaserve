"""Multi-node request router"""

import httpx
import uvicorn
import requests
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

app = FastAPI()

clients = {}
upstreams = []
relations = {}

def build_clients(upstreams: list):
    global clients
    global relations
    for upstream in upstreams:
        client = httpx.AsyncClient(base_url=upstream, timeout=None)
        clients[upstream] = client
        models = get_available_models(upstream)
        for model in models:
            if model not in relations:
                relations[model] = []
            relations[model].append(upstream)

def get_available_models(endpoint: str):
    response = requests.get(f"{endpoint}/v1/models").json()
    models = [x['id'] for x in response['data']]
    return models

def handle_sys_info():
    responses = {}
    for upstream in upstreams:
        response = requests.get(f"{upstream}/sysinfo")
        responses[upstream] = response.json()
    return responses

async def _reverse_proxy(request: Request):
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    if request.url.path == "/sysinfo":
        res =  handle_sys_info()
        return JSONResponse(content=res, status_code=200)
    res = await request.json()
    model = res['model']
    client = clients[relations[model][0]]
    
    rp_req = client.build_request(
        request.method, url, headers=request.headers.raw, content=await request.body()
    )
    rp_resp = await client.send(rp_req, stream=True)
    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose),
    )


app.add_route("/{path:path}", _reverse_proxy, ["GET", "POST"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--upstreams", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    upstreams = args.upstreams.split(",")
    build_clients(upstreams)
    print(relations)
    uvicorn.run(app, host=args.host, port=args.port)