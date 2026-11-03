"""#674 cell 4: rank measured rows against the manifest's objective and
emit the winner's pinned, serve-consumable recommendation.

Input: a tune manifest + the row JSONs `auto_tune_measure_a6000` wrote
(one per candidate). Output: a ranking report (every row's gate verdict,
the eliminated with reasons, the unmeasurable disclosed) plus the winner
record — the version-1 serve model-config manifest `damendaemon serve
--model-config` consumes as-is, alongside the exact environment and flags
to launch it with. Never silently changes anything: the recommendation is
reviewable and falsifiable from the evidence alone.
"""
import argparse
import json
from pathlib import Path

from damendaemon.autotune import (Candidate, Measurement, parse_manifest,
                                   rank, winner_record)
from scripts.experiments.split_gguf import save


def load_rows(paths):
    rows = []
    for path in paths:
        raw = json.loads(Path(path).read_text())
        if not raw.get('completed'):
            raise ValueError(f'row {path} is incomplete; refusing to rank '
                             'on partial evidence (rerun the candidate)')
        rows.append(Measurement(
            candidate=Candidate(**raw['candidate']),
            metric_value=raw.get('metric_value'),
            first_output_seconds=raw.get('first_output_seconds') or [],
            request_seconds=raw.get('request_seconds') or [],
            completion_tokens=raw.get('completion_tokens') or [],
            errors=raw.get('errors') or 0,
            unmeasurable_reason=raw.get('unmeasurable_reason')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--rows', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    manifest = parse_manifest(args.manifest.read_bytes())
    rows = load_rows(args.rows)
    if not rows:
        raise ValueError('no measured rows to rank')
    ranked, eliminated = rank(rows, manifest['objective'])
    report = dict(manifest_model=manifest['model'],
                  objective=manifest['objective'],
                  rows_ranked=[dict(candidate=row.candidate.__dict__,
                                    metric_value=row.metric_value) for row in ranked],
                  rows_eliminated=[dict(candidate=row.candidate.__dict__,
                                        metric_value=row.metric_value,
                                        reasons=reasons) for row, reasons in eliminated],
                  winner=None)
    if ranked:
        winner = ranked[0]
        record = winner_record(manifest, winner.candidate, winner.metric_value)
        report['winner'] = record
        save(args.output, report)
        # The serve-consumable model-config, emitted beside the report.
        save(args.output.with_suffix('.model-config.json'), record['model'])
        print(f"winner: {winner.candidate} metric {winner.metric_value}")
        print(f"serve model-config: {args.output.with_suffix('.model-config.json')}")
    else:
        save(args.output, report)
        raise SystemExit('no candidate passed the declared gates; nothing to recommend')


if __name__ == '__main__':
    main()
