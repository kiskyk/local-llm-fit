# serve.py — static file serving and /api/hf relay for local development
import http.server, socketserver, urllib.parse, urllib.request

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
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
            with urllib.request.urlopen(req) as res:
                body = res.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
    print(f'http://127.0.0.1:{PORT}')
    httpd.serve_forever()
