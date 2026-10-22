from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

# No tracing or cloud model calls in this demo.
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from knowledge import KnowledgeBase, MAX_BYTES
from runtime import OfflineRuntime

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    app.state.kb = KnowledgeBase(Path(os.environ.get('KNOWLEDGE_DATA_DIR', ROOT / 'data')))
    app.state.runtime = OfflineRuntime(app.state.kb)
    try:
        yield
    finally:
        app.state.kb.close()


app = FastAPI(title='企业知识检索 · 离线演示', lifespan=lifespan)


@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html')


@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'offline_extract_only', 'llm_configured': False,
            'documents': len(app.state.kb.documents())}


@app.get('/documents')
def documents():
    return app.state.kb.documents()


@app.post('/ingest')
async def ingest(file: UploadFile = File(...)):
    content = await file.read(MAX_BYTES + 1)
    try:
        return await run_in_threadpool(app.state.kb.ingest, file.filename or '', content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, '文档解析失败，请检查文件是否损坏或加密。') from exc


@app.post('/demo')
def demo():
    return app.state.kb.ingest('示例差旅制度.md', (ROOT / 'sample.md').read_bytes())


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=8)
    topk: int = Field(default=3, ge=1, le=10)
    return_scores: bool = False


@app.post('/retrieve')
def retrieve(request: RetrieveRequest):
    if any(not q.strip() or len(q) > 2000 for q in request.queries):
        raise HTTPException(400, '查询不能为空或超过 2000 字。')
    output = []
    for query in request.queries:
        hits = app.state.kb.search(query, request.topk)
        docs = []
        for hit in hits:
            item = {'document': {'doc_id': hit['chunk_id'], 'parent_document_id': hit['doc_id'],
                                 'title': hit['name'], 'file_name': hit['name'],
                                 'contents': hit['text'], 'text': hit['text'],
                                 'spans': hit['spans'], 'version': hit['version'],
                                 'location': hit['location']}}
            if request.return_scores:
                item['score'] = hit['score']
            docs.append(item)
        output.append(docs)
    return {'result': output}


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.post('/ask')
def ask(request: AskRequest):
    return app.state.runtime.ask(request.question)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765)
