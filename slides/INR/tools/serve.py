"""Serve the talk locally without caching (handy while editing; opening index.html directly also works).

    python tools/serve.py [port]      ->  http://127.0.0.1:8791/
"""
import functools
import http.server
import os
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    handler = functools.partial(NoCache, directory=root)
    print(f'Serving {root} at http://127.0.0.1:{port}/')
    http.server.ThreadingHTTPServer(('127.0.0.1', port), handler).serve_forever()
