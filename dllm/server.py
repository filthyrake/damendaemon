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


def handler_for(runtime, auth=None):
    from .http_auth import AuthPolicy
    auth = AuthPolicy.from_env() if auth is None else auth
    from .client_admission import ClientAdmission, ClientLimitError
    from .http_auth import INFERENCE_ROUTES
    admission = ClientAdmission(auth)
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
            if status == 401:
                self.send_header("WWW-Authenticate", 'Bearer realm="dllm"')
            if status == 429:
                self.send_header('Retry-After', '1')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def authorize(self):
            status = auth.authorize(self.command, self.path, self.headers.get_all('Authorization', []))
            if status is None:
                return True
            self.send_json(status, {'error': 'unauthorized' if status == 401 else 'forbidden'})
            return False

        def do_GET(self):
            if not self.authorize():
                return
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [
                    {"id": alias, "object": "model", "owned_by": "dllm"}
                    for alias in runtime.model_ids()
                ]})
            elif self.path == '/v1/scheduler/status':
                snapshot = getattr(runtime, 'scheduler_status', None)
                if snapshot is None:
                    self.send_json(501, {'error': 'scheduler status is unavailable for this runtime'})
                else:
                    self.send_json(200, snapshot())
            elif self.path == '/v1/models/status':
                try:
                    self.send_json(200, runtime.manage('status'))
                except BusyError:
                    self.send_json(503, {'error': 'model management is busy'})
            elif self.path == '/v1/models/config':
                try:
                    self.send_json(200, runtime.model_manifest())
                except (ValueError, OSError):
                    self.send_json(400, {'error': 'model registry cannot be exported'})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if not self.authorize():
                return
            if self.path in INFERENCE_ROUTES:
                try:
                    with admission.lease(self.headers.get_all('Authorization', [])):
                        self.post_authorized()
                except ClientLimitError:
                    self.send_json(429, {'error': 'client outstanding inference limit reached'})
            else:
                self.post_authorized()

        def post_authorized(self):
            management = {'/v1/models/register': 'register', '/v1/models/load': 'load', '/v1/models/unload': 'unload', '/v1/models/remove': 'remove'}
            if self.path not in ("/v1/scheduler/drain", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/token-scores", "/v1/rerank") and self.path not in management:
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
                if self.path == '/v1/scheduler/drain':
                    if body != {}:
                        raise ValueError('drain requires an empty JSON object')
                    drain = getattr(runtime, 'drain', None)
                    if drain is None:
                        self.send_json(501, {'error': 'scheduler drain is unavailable for this runtime'})
                    else:
                        self.send_json(200, drain())
                    return
                if self.path=='/v1/rerank':
                    if (not isinstance(body,dict) or set(body)-{'model','query','documents','top_n','return_documents'}
                            or not isinstance(body.get('model'),str)):
                        raise ValueError('unsupported reranking request')
                    self.send_json(200,runtime.rerank(body['model'],body.get('query'),body.get('documents'),
                        body.get('top_n'),body.get('return_documents',False)))
                    return
                if self.path=='/v1/token-scores':
                    if (not isinstance(body,dict) or set(body)!={'model','prompt','candidate_token_ids'}
                            or not isinstance(body['model'],str)):
                        raise ValueError('token scoring requires model, prompt and candidate_token_ids')
                    from .scoring import options
                    prompt,candidates=options(body['prompt'],body['candidate_token_ids'])
                    self.send_json(200,runtime.score(body['model'],prompt,candidates))
                    return
                if self.path == '/v1/embeddings':
                    if (not isinstance(body, dict) or set(body) - {'model','input','encoding_format','dimensions'}
                            or not isinstance(body.get('model'), str)):
                        raise ValueError('unsupported embedding request')
                    self.send_json(200, runtime.embed(body['model'], body.get('input'),
                        body.get('encoding_format','float'), body.get('dimensions')))
                    return
                if self.path in management:
                    action = management[self.path]
                    fields = {'model'} | ({'path','quantization','chat_template'} if action == 'register' else {'release_host'} if action == 'unload' else set())
                    if not isinstance(body, dict) or set(body) - fields:
                        raise ValueError('unsupported model management fields')
                    self.send_json(200, runtime.manage(action, **body))
                    return
                allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "stop", "temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias", "stream", "stream_options", "chat_template_kwargs", "response_format"}
                if raw:
                    allowed = (allowed - {'messages', 'chat_template_kwargs', 'max_completion_tokens'}) | {'prompt'}
                else:
                    allowed |= {'logprobs','top_logprobs','tools','tool_choice','reasoning_effort'}
                if not isinstance(body, dict) or set(body) - allowed:
                    raise ValueError("unsupported request fields")
                from .tool_calls import prepare_tools,validate_messages,ToolCallStream,JsonToolCallStream
                selected_tools,tool_schema=prepare_tools(body.get('tools'),body.get('tool_choice'))
                sampling = {key: body[key] for key in ("temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias") if key in body}
                Sampling(**sampling)
                from .logprobs import options as score_options
                report_scores=score_options(body.get('logprobs',False),body.get('top_logprobs',0)) is not None
                if report_scores:
                    sampling.update(logprobs=True,top_logprobs=body.get('top_logprobs') or 0)
                if "response_format" in body:
                    from .structured import response_schema
                    response_schema(body["response_format"])
                    sampling["response_format"] = body["response_format"]
                if 'stop' in body:
                    stops=stop_sequences(body['stop'])
                    if report_scores and stops:
                        raise ValueError('logprobs with custom text stops is not yet supported')
                    sampling['stop'] = body['stop']
                if selected_tools:
                    if report_scores or body.get('response_format') is not None or sampling.get('stop'):
                        raise ValueError('tool calls cannot combine with structured output, scores or custom stops')
                    sampling['tools']=selected_tools
                    sampling['tool_choice']=body.get('tool_choice')
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
                else:
                    from .vision_policy import enabled as vision_enabled
                    if vision_enabled():
                        from .vision_messages import validate_image_messages
                        messages=validate_image_messages(messages)
                    else:
                        messages=validate_messages(messages)
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
                if tool_schema is not None and template_options.get('enable_thinking') is True:
                    raise ValueError('forced tool calls cannot use thinking')
                if 'reasoning_effort' in body:
                    if not isinstance(body['reasoning_effort'],str) or body['reasoning_effort'] not in ('low','medium','high'):
                        raise ValueError('reasoning_effort must be low, medium or high')
                    template_options={**template_options,'reasoning_effort':body['reasoning_effort']}
                stream = body.get("stream", False)
                alias = body["model"]
                request_id = ("cmpl-" if raw else "chatcmpl-") + uuid.uuid4().hex
                created = int(time.time())
                stream_object = 'text_completion' if raw else 'chat.completion.chunk'
                def chunk(delta, finish=None, score=None):
                    choice = {'text': delta.get('content', ''), 'logprobs': None} if raw else {'delta': delta}
                    if report_scores:
                        choice['logprobs']={'content':[score]} if score is not None else None
                    return {"id": request_id, "object": stream_object,
                            "created": created, "model": alias,
                            "choices": [{"index": 0, **choice, "finish_reason": finish}]}
                def send_event(payload):
                    self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                    self.wfile.flush()
                text = ""
                reasoning_text = ""
                tool_parser=(JsonToolCallStream(tool_schema) if tool_schema is not None
                    else ToolCallStream(selected_tools) if selected_tools else None)
                native_calls=[]
                token_scores=[]
                final = None
                with closing(runtime.generate(alias, messages, max_tokens, **template_options, **sampling)) as generation:
                    for event in generation:
                        if event.get("tool_call_format")=="harmony":tool_parser=None
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
                            final = dict(event)
                            if tool_parser:
                                for delta in tool_parser.finish(final['finish_reason']):
                                    text+=delta.get('content','')
                                    if stream:send_event(chunk(delta))
                                if tool_parser.calls and final['finish_reason']=='stop':
                                    final['finish_reason']='tool_calls'
                            if stream:
                                send_event(chunk({}, final["finish_reason"]))
                                if include_usage:
                                    send_event({"id": request_id, "object": stream_object,
                                                "created": created, "model": alias, "choices": [],
                                                "usage": event["usage"], "dllm_timing": event["timing"]})
                        else:
                            score=event.get('logprobs')
                            if score is not None and not stream:
                                token_scores.append(score)
                            reasoning=event.get('reasoning_content','')
                            reasoning_text+=reasoning
                            if stream and reasoning:send_event(chunk({'reasoning_content':reasoning}))
                            deltas=tool_parser.feed(event['text']) if tool_parser else [{'content':event['text']}]
                            if event.get('tool_calls'):
                                native_calls.extend(event['tool_calls'])
                                deltas.append({'tool_calls':[{'index':i,**call} for i,call in enumerate(event['tool_calls'])]})
                            for delta in deltas:
                                text+=delta.get('content','')
                                if stream and (delta.get('content') or delta.get('tool_calls') or score is not None):
                                    send_event(chunk(delta,score=score))
                if stream:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    choice = {'text': text, 'logprobs': None} if raw else {'message': {'role': 'assistant', 'content': text}}
                    if not raw and reasoning_text:choice['message']['reasoning_content']=reasoning_text
                    if tool_parser and tool_parser.calls:
                        choice['message'].update(content=text or None,tool_calls=tool_parser.calls)
                    if native_calls:
                        choice['message'].update(content=text or None,tool_calls=native_calls)
                    if report_scores:
                        choice['logprobs']={'content':token_scores}
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
