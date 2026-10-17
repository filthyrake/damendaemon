"""Small loopback-only development API with bounded inference admission."""

from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import signal
import threading
import traceback
import time
import uuid

from .runtime import BusyError
from .sampling import Sampling
from .stopping import stop_sequences
from .prompts import validate_raw

BODY_READ_SECONDS = 30



def request_json(encoded):
    """Decode unambiguous, finite JSON before dispatching a client request."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate request JSON field')
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('non-finite request JSON number')
        return number

    try:
        return json.loads(encoded, object_pairs_hook=unique_fields,
                          parse_float=finite_float, parse_constant=finite_float)
    except RecursionError as error:
        raise ValueError('request JSON nesting is too deep') from error


def handler_for(runtime, auth=None, *, reservation_store=None):
    from .http_auth import AuthPolicy
    auth = AuthPolicy.from_env() if auth is None else auth
    from .client_admission import ClientAdmission, ClientLimitError, ClientOutputLimitError, ClientRateLimitError, ClientOutputRateLimitError, SharedRateUnavailable, SharedReservationUnavailable
    from .http_auth import INFERENCE_ROUTES
    admission = ClientAdmission(auth,reservation_store=reservation_store)
    from .request_scope import CacheScope, cache_scope
    shared_scope = object()
    client_scopes = {client.name: CacheScope(client.max_prefix_cache_tokens,client.decode_weight,client.prefill_weight) for client in auth.clients}
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_response(self,*args,**kwargs):
            self.response_started=True
            return super().send_response(*args,**kwargs)

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Do not log prompts or response content.

        def read_body(self, length):
            deadline = time.monotonic() + BODY_READ_SECONDS
            previous_timeout = self.connection.gettimeout()
            body = bytearray()
            try:
                while len(body) < length:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('request body deadline exceeded')
                    self.connection.settimeout(remaining if previous_timeout is None
                                               else min(previous_timeout, remaining))
                    # read1 performs at most one underlying read. A buffered
                    # read(length) could keep resetting the inactivity timeout.
                    part = self.rfile.read1(min(65536, length - len(body)))
                    if not part:
                        raise ValueError('incomplete request body')
                    body.extend(part)
                if time.monotonic() >= deadline:
                    raise TimeoutError('request body deadline exceeded')
                return body
            finally:
                self.connection.settimeout(previous_timeout)

        def send_json(self, status, payload, *, retry_after=1):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 401:
                self.send_header("WWW-Authenticate", 'Bearer realm="damendaemon"')
            if status == 429:
                self.send_header('Retry-After', str(retry_after))
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
                health = getattr(runtime, 'is_healthy', None)
                healthy = health is None or health()
                self.send_json(200 if healthy else 503,
                               {"status": "ok" if healthy else "unavailable"})
            elif self.path == "/v1/models":
                client = auth.client(self.headers.get_all('Authorization', []))
                self.send_json(200, {"object": "list", "data": [
                    {"id": alias, "object": "model", "owned_by": "damendaemon"}
                    for alias in runtime.model_ids()
                    if client is None or client.allows_model(alias)
                ]})
            elif self.path == '/v1/scheduler/status':
                snapshot = getattr(runtime, 'scheduler_status', None)
                if snapshot is None:
                    self.send_json(501, {'error': 'scheduler status is unavailable for this runtime'})
                else:
                    self.send_json(200, snapshot())
            elif self.path == '/v1/models/status':
                try:
                    status = runtime.manage('status')
                except BusyError:
                    self.send_json(503, {'error': 'model management is busy'})
                except Exception:
                    self.send_json(500, {'error': 'model management failed'})
                else:
                    self.send_json(200, status)
            elif self.path == '/v1/models/config':
                try:
                    self.send_json(200, runtime.model_manifest())
                except (ValueError, OSError):
                    self.send_json(400, {'error': 'model registry cannot be exported'})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            self.response_started=False
            if not self.authorize():
                return
            if self.path in INFERENCE_ROUTES:
                client = auth.client(self.headers.get_all('Authorization', []))
                scope_token = cache_scope.set(shared_scope if client is None else client_scopes[client.name])
                try:
                    with admission.lease(self.headers.get_all('Authorization', [])):
                        self.post_authorized()
                except SharedReservationUnavailable:
                    if self.response_started:self.close_connection=True
                    else:self.send_json(503, {'error': 'shared client reservation accounting unavailable'})
                except SharedRateUnavailable:
                    self.send_json(503, {'error': 'shared client rate accounting unavailable'})
                except ClientRateLimitError as error:
                    self.send_json(429, {'error': 'client inference request rate limit reached'}, retry_after=error.retry_after)
                except ClientLimitError:
                    self.send_json(429, {'error': 'client outstanding inference limit reached'})
                finally:
                    cache_scope.reset(scope_token)
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
                lengths = self.headers.get_all("Content-Length", [])
                if self.headers.get_all("Transfer-Encoding", []):
                    raise ValueError("Transfer-Encoding is not supported")
                if (len(lengths) != 1 or not lengths[0].isascii()
                        or not lengths[0].isdecimal()):
                    raise ValueError("one decimal Content-Length is required")
                length = int(lengths[0])
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("body must be between 1 byte and 1 MiB")
                encoded_body = self.read_body(length)
                body = request_json(encoded_body)
                if self.path in INFERENCE_ROUTES and isinstance(body, dict) and isinstance(body.get('model'), str):
                    client = auth.client(self.headers.get_all('Authorization', []))
                    if client is not None and not client.allows_model(body['model']):
                        self.send_json(403, {'error': 'forbidden'})
                        return
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
                native_tools=getattr(runtime,'native_tool_calls',False) is True
                prepare=runtime.prepare_tools if native_tools else prepare_tools
                selected_tools,tool_schema=prepare(body.get('tools'),body.get('tool_choice'))
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
                    if native_tools:
                        messages=runtime.prepare_messages(messages,tools=selected_tools)
                    elif vision_enabled():
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
                tool_parser=(None if native_tools else JsonToolCallStream(tool_schema) if tool_schema is not None
                    else ToolCallStream(selected_tools) if selected_tools else None)
                native_calls=[]
                token_scores=[]
                final = None
                with admission.output_lease(self.headers.get_all('Authorization', []), max_tokens), \
                        closing(runtime.generate(alias, messages, max_tokens, **template_options, **sampling)) as generation:
                    for event in generation:
                        if final is not None:
                            raise RuntimeError('generation emitted an event after its terminal event')
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
                            if (not isinstance(event['finish_reason'], str) or not event['finish_reason']
                                    or not isinstance(event.get('usage'), dict)
                                    or not isinstance(event.get('timing'), dict)):
                                raise RuntimeError('invalid generation terminal event')
                            final = dict(event)
                            if tool_parser:
                                for delta in tool_parser.finish(final['finish_reason']):
                                    text+=delta.get('content','')
                                    if stream:send_event(chunk(delta))
                                if tool_parser.calls and final['finish_reason']=='stop':
                                    final['finish_reason']='tool_calls'
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
                if final is None:
                    raise RuntimeError('generation ended without a terminal event')
                if stream:
                    # Generator and output-lease cleanup must succeed before
                    # publishing completion, even when text was already sent.
                    send_event(chunk({}, final["finish_reason"]))
                    if include_usage:
                        send_event({"id": request_id, "object": stream_object,
                                    "created": created, "model": alias, "choices": [],
                                    "usage": final["usage"], "damendaemon_timing": final["timing"]})
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
                        "usage": final["usage"], "damendaemon_timing": final["timing"]})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception as error:
                if isinstance(error, SharedReservationUnavailable):
                    status, message = 503, 'shared client reservation accounting unavailable'
                elif isinstance(error, SharedRateUnavailable):
                    status, message = 503, 'shared client rate accounting unavailable'
                elif isinstance(error, ClientOutputLimitError):
                    status, message = 429, 'client outstanding output token budget exceeded'
                elif isinstance(error, ClientOutputRateLimitError):
                    status, message = 429, 'client declared output token rate limit reached'
                elif isinstance(error, BusyError):
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
                    self.send_json(status, {"error": message}, retry_after=getattr(error, "retry_after", 1))
    Handler.client_admission=admission
    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    """Cap connection threads as well as the runtime's inference queue."""
    def __init__(self, address, handler, max_connections=16):
        try:
            if type(max_connections) is not int or max_connections <= 0:
                raise ValueError('HTTP connection capacity must be a positive integer')
            self.connections = threading.BoundedSemaphore(max_connections)
            super().__init__(address, handler)
        except BaseException:
            admission=getattr(handler,'client_admission',None)
            if admission is not None and admission.reservation_store is not None:admission.close()
            raise

    def server_close(self):
        super().server_close()
        admission=getattr(self.RequestHandlerClass,'client_admission',None)
        if admission is not None and admission.reservation_store is not None:admission.close()

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
    server=None;previous={};primary=None;stopping=threading.Event()
    def interrupt(*_):stopping.set()
    try:
        server=BoundedHTTPServer(("127.0.0.1",port),handler_for(runtime))
        server.daemon_threads=False
        server.block_on_close=True
        server.timeout=.2
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT,signal.SIGTERM):previous[signum]=signal.signal(signum,interrupt)
        print(f"damendaemon listening on http://127.0.0.1:{server.server_port}",flush=True)
        while not stopping.is_set():server.handle_request()
    except KeyboardInterrupt:
        pass
    except BaseException as error:primary=error;raise
    finally:
        errors=[]
        # Stop/wake scheduler consumers before waiting for their HTTP threads.
        if hasattr(runtime,'close'):
            try:runtime.close()
            except BaseException as error:errors.append(error)
        if server is not None:
            try:server.server_close()
            except BaseException as error:errors.append(error)
        for signum,handler in previous.items():signal.signal(signum,handler)
        if errors:
            if primary is not None:primary.add_note('HTTP shutdown cleanup failed; retain owners for retry')
            else:raise errors[0]
