"""
Jarvis4 HTTP API — endpoints for text chat, TTS, and alerts.

Endpoints:
  GET  /status  — health check
  POST /chat    — send text to LLM, optionally speak response
  POST /speak   — direct TTS (no LLM)
  POST /alert   — urgent message, always spoken
"""

import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger("jarvis4.http")


class Jarvis4HTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Jarvis4 API."""

    # Set by the entry point before starting the server
    llm_client = None
    tts_client = None
    state_queue = None  # queue.Queue for state-machine routing

    def log_message(self, format, *args):
        log.debug(format, *args)

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _enqueue_speak_request(self, action, text):
        """Push a request onto the state-machine queue and wait."""
        done_event = threading.Event()
        result = {}
        req = {"action": action, "text": text, "done": done_event, "result": result}
        self.__class__.state_queue.put(req)
        done_event.wait(timeout=120)
        return result

    def do_GET(self):
        if self.path == "/status":
            self._send_json(200, {"status": "ok", "service": "jarvis4"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        body = self._read_body()
        text = body.get("text", body.get("message", "")).strip()

        if self.path == "/chat":
            if not text:
                self._send_json(400, {"error": "missing text"})
                return
            speak = body.get("speak", False)
            if speak:
                result = self._enqueue_speak_request("chat", text)
                self._send_json(200, {"response": result.get("response", "")})
            else:
                llm = self.__class__.llm_client
                if not llm:
                    self._send_json(503, {"error": "not initialized"})
                    return
                response = llm.chat(text)
                self._send_json(200, {"response": response or ""})

        elif self.path == "/speak":
            if not text:
                self._send_json(400, {"error": "missing text"})
                return
            result = self._enqueue_speak_request("speak", text)
            if result.get("error"):
                self._send_json(500, {"error": result["error"]})
            else:
                self._send_json(200, {"status": "spoken", "text": text})

        elif self.path == "/alert":
            if not text:
                self._send_json(400, {"error": "missing text"})
                return
            result = self._enqueue_speak_request("alert", text)
            self._send_json(200, {"response": result.get("response", "")})

        else:
            self._send_json(404, {"error": "not found"})


def start_http_server(port: int, llm_client, tts_client, state_queue) -> HTTPServer:
    """Start the HTTP API server in a background thread."""
    Jarvis4HTTPHandler.llm_client = llm_client
    Jarvis4HTTPHandler.tts_client = tts_client
    Jarvis4HTTPHandler.state_queue = state_queue

    httpd = HTTPServer(("0.0.0.0", port), Jarvis4HTTPHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    log.info("HTTP API: http://0.0.0.0:%d", port)
    return httpd
