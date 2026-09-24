"""Serve the demo locally: public/ as static files and /api/extract through the real handler.

    python -m scripts.dev_server   # http://localhost:3004
"""
from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from api.extract import handler as ApiHandler

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")


class Dev(SimpleHTTPRequestHandler):
    _send = ApiHandler._send

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        if self.path.startswith("/api/extract"):
            return ApiHandler.do_GET(self)
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/extract"):
            return ApiHandler.do_POST(self)
        self.send_error(404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3004"))
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("", port), Dev).serve_forever()
