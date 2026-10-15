# Flash shared-core regression and repair (#241)

The full local suite at the placement boundary `7e83257` ran 1,758 tests in
478.023 seconds and failed six tests. Four image-generation/tool tests and two
image-HTTP tests failed from the same cause: shared Flash input validators began
reading constructor-only `self.device`, while the existing scripted image models
initialize actual embedding storage without invoking that constructor.

Repair `bab945d` validates against `self.embedding.device`. This retains exact
same-device admission and preserves the established subclass contract. No
numerical tolerance, resource limit or timeout changed. All 26 focused image,
HTTP, model, feature and CUDA-admission regressions pass after the repair.
The complete repaired-suite result is pending; this document does not claim it
passed. Tiny CUDA arithmetic/command evidence remains separately scoped.
