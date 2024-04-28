import httpx
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post('/{path:path}')
async def reverse_proxy(request: Request):
    client = httpx.AsyncClient(base_url="http://127.0.0.1:8000/", timeout=12000)
    url = httpx.URL(path=request.url.path, query=request.url.query.encode('utf-8'))
    print(url)
    req = client.build_request(
        request.method, url, headers=request.headers.raw, content=request.stream()
    )
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        status_code=r.status_code,
        headers=r.headers,
        background=BackgroundTask(r.aclose)
    )
