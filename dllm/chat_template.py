"""Shared sandboxed chat templates and explicit per-model overrides."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path

from jinja2 import Template, TemplateError, nodes
from jinja2.visitor import NodeTransformer
from jinja2.sandbox import ImmutableSandboxedEnvironment

MAX_TEMPLATE_BYTES = 128 * 1024
MAX_TEMPLATE_ITERATIONS = 1_000_000
_LOOP_BUDGET = ContextVar('dllm_template_loop_budget', default=None)
# Jinja expressions can raise Python type/arithmetic errors outside TemplateError.
# Keep infrastructure RuntimeError/OSError outside request-error normalization.
RENDER_ERRORS = (TemplateError, RecursionError, TypeError, ArithmeticError)


def _budget():
    current = _LOOP_BUDGET.get()
    return [MAX_TEMPLATE_ITERATIONS] if current is None else current


@contextmanager
def _budget_scope(budget):
    token = _LOOP_BUDGET.set(budget)
    try:
        yield
    finally:
        _LOOP_BUDGET.reset(token)


class _BoundedTemplate(Template):
    def render(self, *args, **kwargs):
        with _budget_scope(_budget()):
            return super().render(*args, **kwargs)

    def generate(self, *args, **kwargs):
        budget = _budget()
        source = super().generate(*args, **kwargs)
        try:
            while True:
                # Do not leave request context installed while yielding to a
                # caller, which may interleave another render or change threads.
                with _budget_scope(budget):
                    try:
                        piece = next(source)
                    except StopIteration:
                        return
                yield piece
        finally:
            with _budget_scope(budget):
                source.close()


class _BoundedLoops(NodeTransformer):
    def visit_For(self, node, *args, **kwargs):
        node = self.generic_visit(node, *args, **kwargs)
        step = nodes.Call(nodes.EnvironmentAttribute('_loop_step'), [], [], None, None)
        step.set_lineno(node.lineno)
        if node.test is not None:
            # Count candidates rejected by the filter, including recursive
            # loops; counting only their bodies would leave silent work unbound.
            node.test = nodes.And(step, node.test).set_lineno(node.lineno)
        else:
            node.body.insert(0, nodes.ExprStmt(step).set_lineno(node.lineno))
        return node


class _BoundedEnvironment(ImmutableSandboxedEnvironment):
    template_class = _BoundedTemplate

    def _parse(self, source, name, filename):
        return _BoundedLoops().visit(super()._parse(source, name, filename))

    def _loop_step(self):
        budget = _LOOP_BUDGET.get()
        if budget is None:
            raise TemplateError('chat template loop iteration budget is unavailable')
        budget[0] -= 1
        if budget[0] < 0:
            raise TemplateError('chat template loop iteration budget exceeded')
        return True


def compile_template(text):
    environment = _BoundedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                      extensions=['jinja2.ext.loopcontrols'])
    # Preserve tokenizer JSON semantics, including insertion order and Unicode.
    environment.filters['tojson'] = partial(json.dumps, ensure_ascii=False, allow_nan=False)
    def raise_exception(message):
        raise ValueError(message)
    environment.globals.update(raise_exception=raise_exception,
                               strftime_now=lambda fmt: datetime.now().strftime(fmt))
    return environment.from_string(text)


def validate_override(text, config=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('chat_template must be nonempty template text')
    try:
        size = len(text.encode('utf-8'))
    except UnicodeError as error:
        raise ValueError('chat_template must be valid UTF-8') from error
    if size > MAX_TEMPLATE_BYTES or '\x00' in text:
        raise ValueError('chat_template must be at most 128 KiB and contain no NUL')
    try:
        compile_template(text)
    except (TemplateError, RecursionError) as error:
        raise ValueError('invalid chat_template syntax') from error
    if config is not None and (config.get('task') in ('embedding', 'rerank') or
            config.get('model_type') in ('mamba', 'mamba2', 't5')):
        raise ValueError('chat_template overrides require a chat-capable decoder')
    return text


def read_override(path):
    with Path(path).expanduser().open('rb') as file:
        data = file.read(MAX_TEMPLATE_BYTES + 1)
    if len(data) > MAX_TEMPLATE_BYTES:
        raise ValueError('chat_template must be at most 128 KiB')
    return validate_override(data.decode('utf-8'))


def override_metadata(text):
    return {'source': 'checkpoint' if text is None else 'override',
            'sha256': None if text is None else hashlib.sha256(text.encode('utf-8')).hexdigest()}
