"""Small loopback-only development API with bounded inference admission."""

from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import traceback
import time
import uuid

from .runtime import BusyError
from .sampling import Sampling
from .stopping import stop_sequences
from .prompts import validate_raw


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Do not log prompts or response content.

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [
                    {"id": alias, "object": "model", "owned_by": "dllm"}
                    for alias in runtime.model_ids()
                ]})
            elif self.path == '/v1/models/status':
                try:
                    self.send_json(200, runtime.manage('status'))
                except BusyError:
                    self.send_json(503, {'error': 'model management is busy'})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            management = {'/v1/models/register': 'register', '/v1/models/load': 'load', '/v1/models/unload': 'unload'}
            if self.path not in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings") and self.path not in management:
                self.send_json(404, {"error": "not found"})
                return
            started = False
            raw = self.path == '/v1/completions'
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("body must be between 1 byte and 1 MiB")
                body = json.loads(self.rfile.read(length))
                if self.path == '/v1/embeddings':
                    if (not isinstance(body, dict) or set(body) - {'model','input','encoding_format','dimensions'}
                            or not isinstance(body.get('model'), str)):
                        raise ValueError('unsupported embedding request')
                    self.send_json(200, runtime.embed(body['model'], body.get('input'),
                        body.get('encoding_format','float'), body.get('dimensions')))
                    return
                if self.path in management:
                    action = management[self.path]
                    fields = {'model'} | ({'path'} if action == 'register' else {'release_host'} if action == 'unload' else set())
                    if not isinstance(body, dict) or set(body) - fields:
                        raise ValueError('unsupported model management fields')
                    self.send_json(200, runtime.manage(action, **body))
                    return
                allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "stop", "temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias", "stream", "stream_options", "chat_template_kwargs", "response_format"}
                if raw:
                    allowed = (allowed - {'messages', 'chat_template_kwargs', 'max_completion_tokens'}) | {'prompt'}
                if not isinstance(body, dict) or set(body) - allowed:
                    raise ValueError("unsupported request fields")
                sampling = {key: body[key] for key in ("temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias") if key in body}
                Sampling(**sampling)
                if "response_format" in body:
                    from .structured import response_schema
                    response_schema(body["response_format"])
                    sampling["response_format"] = body["response_format"]
                if 'stop' in body:
                    stop_sequences(body['stop'])
                    sampling['stop'] = body['stop']
                if 'max_tokens' in body and 'max_completion_tokens' in body:
                    raise ValueError('specify only one completion token limit')
                max_tokens = body.get('max_completion_tokens', body.get('max_tokens', 128))
                if type(max_tokens) is not int or max_tokens <= 0:
                    raise ValueError('completion token limit must be a positive integer')
                if not isinstance(body.get("model"), str):
                    raise ValueError("model must be a string")
                if "stream" in body and type(body["stream"]) is not bool:
                    raise ValueError("stream must be a boolean")
                messages = body.get("messages")
                if raw:
                    sampling['raw_prompt'] = validate_raw(body.get('prompt'))
                elif not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or set(m) != {"role", "content"}
                    or m["role"] not in ("system", "user", "assistant")
                    or not isinstance(m["content"], str) for m in messages
                ):
                    raise ValueError("messages must contain text role/content objects")
                options = body.get("stream_options", {})
                if not isinstance(options, dict) or set(options) - {"include_usage"}:
                    raise ValueError("unsupported stream options")
                include_usage = options.get("include_usage", False)
                if type(include_usage) is not bool:
                    raise ValueError("include_usage must be boolean")
                template_options = body.get("chat_template_kwargs", {})
                if (not isinstance(template_options, dict)
                        or set(template_options) - {"enable_thinking"}
                        or any(type(value) is not bool for value in template_options.values())):
                    raise ValueError("chat_template_kwargs supports only boolean enable_thinking")
                stream = body.get("stream", False)
                alias = body["model"]
                request_id = ("cmpl-" if raw else "chatcmpl-") + uuid.uuid4().hex
                created = int(time.time())
                stream_object = 'text_completion' if raw else 'chat.completion.chunk'
                def chunk(delta, finish=None):
                    choice = {'text': delta.get('content', ''), 'logprobs': None} if raw else {'delta': delta}
                    return {"id": request_id, "object": stream_object,
                            "created": created, "model": alias,
                            "choices": [{"index": 0, **choice, "finish_reason": finish}]}
                def send_event(payload):
                    self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                    self.wfile.flush()
                text = ""
                final = None
                with closing(runtime.generate(alias, messages, max_tokens, **template_options, **sampling)) as generation:
                    for event in generation:
                        if stream and not started:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.close_connection = True
                            started = True
                            if not raw:
                                send_event(chunk({"role": "assistant"}))
                        if "finish_reason" in event:
                            final = event
                            if stream:
                                send_event(chunk({}, event["finish_reason"]))
                                if include_usage:
                                    send_event({"id": request_id, "object": stream_object,
                                                "created": created, "model": alias, "choices": [],
                                                "usage": event["usage"]})
                        else:
                            text += event["text"]
                            if stream and event["text"]:
                                send_event(chunk({"content": event["text"]}))
                if stream:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    choice = {'text': text, 'logprobs': None} if raw else {'message': {'role': 'assistant', 'content': text}}
                    self.send_json(200, {"id": request_id, "object": "text_completion" if raw else "chat.completion",
                        "created": created, "model": alias,
                        "choices": [{"index": 0, **choice,
                                     "finish_reason": final["finish_reason"]}],
                        "usage": final["usage"], "dllm_timing": final["timing"]})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception as error:
                if isinstance(error, BusyError):
                    status, message = 503, str(error)
                elif isinstance(error, (ValueError, KeyError, TypeError)):
                    status, message = 400, "invalid or unsupported request"
                else:
                    status, message = 500, "model execution failed"
                    traceback.print_exc()
                if started:
                    try:
                        self.wfile.write(("data: " + json.dumps({"error": message}) + "\n\n").encode())
                        self.wfile.flush()
                    except OSError:
                        pass
                    self.close_connection = True
                else:
                    self.send_json(status, {"error": message})
    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    """Cap connection threads as well as the runtime's inference queue."""
    def __init__(self, address, handler, max_connections=16):
        self.connections = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def serve(runtime, port):
    server = BoundedHTTPServer(("127.0.0.1", port), handler_for(runtime))
    print(f"dllm listening on http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(runtime, "close"):
            runtime.close()
        server.server_close()
