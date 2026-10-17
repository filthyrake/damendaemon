"""Strict evidence-state evaluation, not a benchmark runner or issue tracker.

Records are explicit attestations linked to immutable report bytes. Validation
cannot establish that an experiment actually happened or that its reports are
honest; it prevents missing/incomparable/proxy data from becoming a verdict.
"""
import hashlib
import sys
from pathlib import Path
import re

PINS=('model_repository','model_revision','checkpoint_sha256','tokenizer_sha256','precision','hardware','topology',
      'context','prompt_tokens','output_tokens','concurrency','prompts_sha256','eos_policy')
METRICS=('decode_tokens_per_second','aggregate_tokens_per_second','first_output_p50_seconds','first_output_p95_seconds',
         'request_p50_seconds','request_p95_seconds','peak_allocated_bytes','peak_reserved_bytes',
         'cold_start_seconds','shape_compile_seconds','host_restore_seconds','disk_reload_seconds')


def _mapping(value,required,optional=()):
    if not isinstance(value,dict) or set(value)-set(required)-set(optional) or not set(required)<=set(value):
        raise ValueError('scorecard fields are missing or unsupported')


def _text(value,limit=1024):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:raise ValueError('invalid scorecard text')


def _digest(value,length):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is None:raise ValueError('invalid immutable scorecard pin')


def _refs(value):
    if not isinstance(value,list) or not 1<=len(value)<=32:raise ValueError('scorecard evidence requires 1..32 report references')
    for item in value:
        _mapping(item,('path','sha256'));_text(item['path']);_digest(item['sha256'],64)
        path=Path(item['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in item['path']:raise ValueError('scorecard evidence paths must stay relative')


def _identity(value):
    _mapping(value,PINS)
    for name in ('model_repository','precision','hardware','topology','eos_policy'):
        if value[name] is not None:_text(value[name])
    if value['model_revision'] is not None:_digest(value['model_revision'],40)
    for name in ('checkpoint_sha256','tokenizer_sha256','prompts_sha256'):
        if value[name] is not None:_digest(value[name],64)
    for name in ('context','prompt_tokens','output_tokens','concurrency'):
        if value[name] is not None and (type(value[name]) is not int or not 1<=value[name]<=2**20):
            raise ValueError('invalid scorecard workload dimension')
    if all(value[name] is not None for name in ('context','prompt_tokens','output_tokens')):
        if value['prompt_tokens']+value['output_tokens']>value['context']:raise ValueError('scorecard workload exceeds context')


def _gate(record,method,identity):
    if record is None:return None,'missing evidence'
    _mapping(record,('result','scope','method','identity','evidence'));_refs(record['evidence']);_identity(record['identity'])
    if record['result'] not in ('pass','fail') or record['scope'] not in ('actual_hardware','cpu_fixture'):
        raise ValueError('invalid scorecard gate result/scope')
    _text(record['method'])
    if any(value is None for value in identity.values()):return None,'cell identity has unresolved pins'
    if record['identity']!=identity:return None,'gate identity does not match cell'
    if record['scope']!='actual_hardware' or record['method']!=method:return None,'gate lacks required hardware/method evidence'
    return record['result']=='pass',None if record['result']=='pass' else 'reported gate failure'


def _measurement(record,identity,phases):
    if record is None:return None,['missing measurement']
    _mapping(record,('identity','engine_revision','scope','workload_passed','shapes_warmed','decode_timing','warmup_rounds','measured_rounds','metrics','evidence'))
    _identity(record['identity']);_digest(record['engine_revision'],40);_refs(record['evidence']);_text(record['decode_timing'])
    if record['scope'] not in ('actual_hardware','cpu_fixture') or type(record['workload_passed']) is not bool or type(record['shapes_warmed']) is not bool:
        raise ValueError('invalid measurement scope/result')
    for name in ('warmup_rounds','measured_rounds'):
        _mapping(record[name],phases)
        if any(type(n) is not int or not 0<=n<=1000000 for n in record[name].values()):raise ValueError('invalid measured phase counts')
    if not isinstance(record['metrics'],dict) or set(record['metrics'])-set(METRICS):raise ValueError('unknown scorecard metric')
    for name,value in record['metrics'].items():
        if value is None:continue
        if type(value) not in (int,float) or not 0<=value<=sys.float_info.max:raise ValueError('invalid scorecard metric value')
        if name in METRICS[:6] and value<=0:raise ValueError('throughput and measured latency must be positive')
    missing=[name for name in PINS if record['identity'][name]!=identity[name]]
    reasons=['incomparable identity: '+','.join(missing)] if missing else []
    unresolved=[name for name in PINS if identity[name] is None or record['identity'][name] is None]
    if unresolved:reasons.append('unresolved identity: '+','.join(unresolved))
    if record['scope']!='actual_hardware':reasons.append('CPU fixture is not performance qualification')
    if not record['workload_passed']:reasons.append('measurement workload failed')
    if not record['shapes_warmed']:reasons.append('reachable execution shapes were not all warmed')
    if record['decode_timing']!='engine_token_steps':reasons.append('decode metric is absent or a client proxy')
    if any(n<3 for n in record['warmup_rounds'].values()) or any(n<10 for n in record['measured_rounds'].values()):
        reasons.append('insufficient warmed/measured ABBA rounds')
    absent=[name for name in METRICS if record['metrics'].get(name) is None]
    if absent:reasons.append('missing metrics: '+','.join(absent))
    metrics=record['metrics']
    for lower,upper in (('first_output_p50_seconds','first_output_p95_seconds'),('request_p50_seconds','request_p95_seconds'),('peak_allocated_bytes','peak_reserved_bytes')):
        if metrics.get(lower) is not None and metrics.get(upper) is not None and metrics[lower]>metrics[upper]:
            raise ValueError('scorecard metric ordering is inconsistent')
    return (metrics if not reasons else None),reasons


def evaluate_scorecard(document):
    _mapping(document,('version','cells'))
    if type(document['version']) is not int or document['version']!=1 or not isinstance(document['cells'],list) or not 1<=len(document['cells'])<=512:
        raise ValueError('scorecard requires version 1 and 1..512 cells')
    results=[];seen=set()
    for cell in document['cells']:
        _mapping(cell,('id','issue','availability','identity','comparators'),('reason','implementation_commit','availability_evidence','numerical','lifecycle','measurement'))
        _text(cell['id'],128);_identity(cell['identity'])
        if cell['id'] in seen:raise ValueError('duplicate scorecard cell ID')
        seen.add(cell['id'])
        if not isinstance(cell['issue'],str) or re.fullmatch(r'https://github\.com/filthyrake/damendaemon/issues/[1-9][0-9]*',cell['issue']) is None:
            raise ValueError('scorecard cells must link a repository issue URL')
        availability=cell['availability']
        if availability not in ('implemented','unsupported','blocked'):raise ValueError('invalid scorecard availability')
        if 'implementation_commit' in cell:_digest(cell['implementation_commit'],40)
        if 'reason' in cell:_text(cell['reason'])
        if 'availability_evidence' in cell:_refs(cell['availability_evidence'])
        if availability=='implemented':_digest(cell.get('implementation_commit'),40)
        else:
            _text(cell.get('reason'));_refs(cell.get('availability_evidence'))
        numerical,note_n=_gate(cell.get('numerical'),'independent_reference',cell['identity'])
        lifecycle,note_l=_gate(cell.get('lifecycle'),'endpoint_lifecycle',cell['identity'])
        metrics,notes=_measurement(cell.get('measurement'),cell['identity'],('A1','A2'))
        if not isinstance(cell['comparators'],list) or len(cell['comparators'])>16:raise ValueError('invalid comparator list')
        baselines=[];comparators=[];engines=set();complete=True
        for baseline in cell['comparators']:
            _mapping(baseline,('engine','availability'),('reason','measurement','evidence','exact_text_match'))
            _text(baseline['engine'],128)
            if baseline['engine'] in engines:raise ValueError('duplicate comparator engine')
            engines.add(baseline['engine']);state=baseline['availability']
            if state not in ('measured','unsupported','blocked'):raise ValueError('invalid comparator availability')
            if state!='measured':
                if 'measurement' in baseline or 'exact_text_match' in baseline:raise ValueError('unsupported comparator cannot include measured results')
                _text(baseline.get('reason'));_refs(baseline.get('evidence'))
                comparators.append(dict(engine=baseline['engine'],comparable=False,reason=baseline['reason']))
                if state=='blocked':complete=False
                continue
            if 'reason' in baseline:_text(baseline['reason'])
            if 'evidence' in baseline:_refs(baseline['evidence'])
            exact=baseline.get('exact_text_match')
            if exact is not None and type(exact) is not bool:raise ValueError('exact text comparison must be boolean or null')
            values,reasons=_measurement(baseline.get('measurement'),cell['identity'],('B1','B2'))
            if exact is not True:
                complete=False;reasons.append('exact cross-engine text comparison missing or failed')
            comparators.append(dict(engine=baseline['engine'],comparable=values is not None,reasons=reasons,exact_text_match=exact))
            if values is None:complete=False
            else:baselines.append(values)
        verdict=None;checks={}
        if metrics is not None and baselines and complete and numerical is True and lifecycle is True:
            for key in ('decode_tokens_per_second','aggregate_tokens_per_second'):
                checks[key]=metrics[key]>=.95*max(b[key] for b in baselines)
            for key in ('first_output_p50_seconds','first_output_p95_seconds'):
                checks[key]=metrics[key]<=1.10*min(b[key] for b in baselines)
            verdict=all(checks.values())
        state='implemented_unvalidated'
        if numerical is True:state='numerical_pass'
        if numerical is True and lifecycle is True:state='lifecycle_pass'
        if metrics is not None:state='performance_measured'
        if verdict is True:state='competitive_pass'
        if availability!='implemented':state=availability;verdict=None;checks={}
        results.append(dict(id=cell['id'],issue=cell['issue'],state=state,identity=cell['identity'],
                            numerical_pass=numerical,lifecycle_pass=lifecycle,competitive_pass=verdict,
                            gate_notes=dict(numerical=note_n,lifecycle=note_l,measurement=notes),
                            competitive_checks=checks,comparators=comparators,reason=cell.get('reason'),
                            implementation_url=('https://github.com/filthyrake/damendaemon/commit/'+cell['implementation_commit'] if cell.get('implementation_commit') else None),
                            evidence=cell))
    return dict(version=1,scope='qualification_evidence_attestations',cells=results,
                all_cells_competitive=all(row['competitive_pass'] is True for row in results),
                limitation='Report hashes bind evidence bytes, not experiment authenticity. No benchmark is executed; missing data is not a pass.')


def verify_evidence(document,root):
    """Verify every declared report hash under the selected local evidence root."""
    root=Path(root).resolve();checked={};total_bytes=0
    def visit(value):
        nonlocal total_bytes
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('evidence','availability_evidence'):
                    _refs(item)
                    for ref in item:
                        path=(root/ref['path']).resolve()
                        if not path.is_relative_to(root):raise ValueError('scorecard evidence escapes root')
                        if path not in checked:
                            if len(checked)>=256 or not path.is_file():raise ValueError('scorecard evidence requires at most 256 regular files')
                            limit=min(16*1024**2,64*1024**2-total_bytes)
                            if limit<1:raise ValueError('scorecard evidence exceeds aggregate byte limit')
                            with path.open('rb') as reader:raw=reader.read(limit+1)
                            if len(raw)>limit:raise ValueError('scorecard evidence exceeds file or aggregate byte limit')
                            total_bytes+=len(raw)
                            checked[path]=hashlib.sha256(raw).hexdigest()
                        if checked[path]!=ref['sha256']:raise ValueError('scorecard evidence digest mismatch')
                else:visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(document)
    return len(checked)
