# damendaemon

Performance is a product requirement. Never claim GPU performance from mocked
tests or CPU-only runs. Compare identical model revisions, precision, context,
output lengths, and hardware; report latency and throughput separately.
Prioritize working capability coverage at vaguely reasonable speed before peak
performance tuning. Bound validation work to what unblocks the next capability
gate; do not let speculative kernel tuning displace context, quantization,
multimodal, multi-GPU/hybrid or available-hardware support.

Use the repository's GitHub Issues for bugs, tasks, and follow-up work. Inspect
existing issues before creating new ones. Markdown documentation is supporting
evidence, not an issue tracker. Do not install or use an alternative tracker.

For this project, commit and push completed, verified work to its existing
GitHub branch. The user explicitly requires that work exist on the remote;
local-only commits are not delivery. This project instruction overrides any
general preference to keep commits local. Do not wait for another push request.

Keep issue #3 as the current status and ordered roadmap. Before implementation
or a GPU experiment, identify a focused issue with its current state, acceptance
gate, hardware, hypothesis and stopping condition. Use `status:active` for the
one current implementation task, `status:next` for the next bounded task, and
queued/deferred labels for the backlog; umbrella issues use `type:epic`.
Use `status:validating` for a bounded GPU qualification while independent local
implementation advances. Keep at most one GPU experiment running; do not stage
new code on that host or disturb its measurements while the run is active.
After a result or change in direction, update the issue body and roadmap with
commit/evidence links and remaining gaps. Comments preserve history but must
not be the only place current status is recorded. Close completed bounded
tasks without closing broader, unqualified parent goals.
Keep the linked private GitHub project (filthyrake project 3) populated with
repository issues and its Status field aligned with issue workflow labels.
Add new issues to the project; use Done for closed issues and Epic for umbrella
contracts unless an explicit workflow label marks their current execution order.
The project is a view of GitHub Issues, not a separate task tracker.

Keep upstream reference repositories unchanged unless explicitly requested.
Preserve licenses and attribution when incorporating upstream code.

Before any benchmark or sustained inference on HOMER, run the existing
`/home/damen/fan-inference.sh on`; `strix-fan-daemon.service` is a graded curve
and is NOT a substitute (its ramp is too slow for these workloads). Temporarily
stop the curve daemon while the max-fan script owns the controller, then restore
`fan-inference.sh off` and restart the daemon after testing. Do not change its
implementation, boot enablement, or power limits separately. Verify all three
EC fans report `fixed`, level `5`, and elevated live RPM before load. Use
`scripts.experiments.guard_homer` for owned experiments: it fails closed on fan
readback/RPM or CPU/GPU sensor failure, starts only below the cooldown threshold,
and stops the owned process at its conservative thermal cutoff. Keep allocator,
host-memory, no-swap scope and time bounds as well. A previous run hard-crashed
thermally when this prerequisite was missed; memory guards alone are insufficient.
