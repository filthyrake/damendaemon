# dllm

Performance is a product requirement. Never claim GPU performance from mocked
tests or CPU-only runs. Compare identical model revisions, precision, context,
output lengths, and hardware; report latency and throughput separately.

Use the repository's GitHub Issues for bugs, tasks, and follow-up work. Inspect
existing issues before creating new ones. Markdown documentation is supporting
evidence, not an issue tracker. Do not install or use an alternative tracker.

Keep issue #3 as the current status and ordered roadmap. Before implementation
or a GPU experiment, identify a focused issue with its current state, acceptance
gate, hardware, hypothesis and stopping condition. Use `status:active` for the
one current implementation task, `status:next` for the next bounded task, and
queued/deferred labels for the backlog; umbrella issues use `type:epic`.
After a result or change in direction, update the issue body and roadmap with
commit/evidence links and remaining gaps. Comments preserve history but must
not be the only place current status is recorded. Close completed bounded
tasks without closing broader, unqualified parent goals.

Keep upstream reference repositories unchanged unless explicitly requested.
Preserve licenses and attribution when incorporating upstream code.
