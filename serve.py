# serve.py — static file serving and /api/hf relay for local development
import functools
import http.server, os, socketserver, urllib.parse, urllib.request

PORT = 8772
ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    # Cache-Controlを返さないとブラウザのヒューリスティックキャッシュで編集が反映されない
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/hf':
            params = urllib.parse.parse_qs(parsed.query)
            path = params.get('path', [''])[0]
            if not path.startswith('/api/models'):
                self.send_error(400, 'invalid path')
                return
            req = urllib.request.Request(
                'https://huggingface.co' + path,
                headers={'User-Agent': 'local-llm-fit/0.1'},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as res:
                    body = res.read()
            except Exception:
                self.send_error(502, 'upstream error')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

# シングルスレッドだと中継1本の詰まりで全リクエストが固まるため、スレッド化する
class ThreadingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

with ThreadingServer(('127.0.0.1', PORT), functools.partial(Handler, directory=ROOT)) as httpd:
    print(f'http://127.0.0.1:{PORT}')
    httpd.serve_forever()
