"""Bounded real HTTP server smoke test. Always stops and joins its own server."""
import json
import os
import socket
import tempfile
import threading
import urllib.request
from pathlib import Path

import uvicorn


def main():
    with tempfile.TemporaryDirectory() as folder:
        os.environ['KNOWLEDGE_DATA_DIR'] = str(Path(folder) / 'data')
        from app import app
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
        ready = threading.Event()
        old_startup = server.startup

        async def startup(sockets=None):
            await old_startup(sockets=sockets)
            ready.set()

        server.startup = startup
        thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        thread.start()
        try:
            if not ready.wait(20):
                raise RuntimeError('Server startup timeout')
            base = f'http://127.0.0.1:{port}'
            def request(path, data=None):
                req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.load(response)
            assert request('/health')['status'] == 'ok'
            request('/demo', {})
            answer = request('/ask', {'question': '差旅报销需要什么材料？'})
            assert answer['gate']['ok'] and answer['citations']
            result = request('/retrieve', {'queries': ['差旅报销'], 'topk': 3})
            assert result['result'][0][0]['document']['spans']
            print(json.dumps({'real_http_smoke': 'passed', 'gate': answer['gate'],
                              'citations': len(answer['citations']), 'server_stopped_after_test': True}, ensure_ascii=False))
        finally:
            server.should_exit = True
            thread.join(10)
            sock.close()
            if thread.is_alive():
                raise RuntimeError('Server failed to stop')


if __name__ == '__main__':
    main()
