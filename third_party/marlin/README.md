# Marlin kernels

These Apache-2.0 sources come from vLLM commit
`84bcbc62644356270aaaa5e2d0237d03adc9bb3a`, whose Marlin implementation is
adapted from IST-DASLab/Marlin by Elias Frantar and modified by Neural Magic
and vLLM contributors. Their original headers and license are retained.
These third-party files remain under their original license, independently
of damendaemon's project license.

`provenance.json` records original paths, hashes and the limited adaptations.
The compute kernel headers are unchanged. The scalar-type header uses a tiny
standalone host-check adapter; the AWQ repacker retains only its device kernel
and parameterizes its namespace. No Torch stable API bindings, vLLM model
executor, scheduler or runtime are included.

The surrounding damendaemon C interfaces provide optional AWQ and Q4_0 backends.
Their supported scopes, build options and qualification status are documented
in [quantization](../../docs/quantization.md). Additional adapters under
`scripts/experiments` remain isolated experiments. Each change requires
independent packed-format/matrix/model gates and real performance/lifecycle
checks before normal dispatch selects it; these headers alone do not establish
runtime support or numerical equivalence.
