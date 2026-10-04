# NInfer and Jev research

Inspected 2026-10-04 for #33 and the architecture/performance roadmap #3.
No upstream runtime dependency or source code has been added by this research.

| Source | Pin / license | Useful direction | Applicability to dllm |
|---|---|---|---|
| [NInfer](https://github.com/Neroued/ninfer) | `c772812b7f6f0cf893663b19971f49b6d9ce0e6e`, Apache-2.0 | Indexed proposal output heads; speculative verification; recurrent-state replay; cache residency accounting | Current upstream explicitly targets RTX 5090 / sm_120a. Kernel performance cannot be assumed on A2. Algorithms can inform independent implementations. |
| [TypeSafe Jev](https://docs.typesafe.ai/primitives/choice.md) | Official public interface documentation, accessed 2026-10-04 | Typed choice/score/yes-no answers without a generated answer string | An interface and architectural idea. This inspection did not provide proprietary model weights or training code to port. Raw token probabilities do not reproduce trained or calibrated decisions. |
| [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) | `49935ef8a32c798701dbff43dd4ad6888698714b`, MIT | Learned decision head; shared context/question/candidate trees; fused final normalization and scoring | An independent project, not TypeSafe's implementation. Its trained-head and tree behavior require separate checkpoint and mathematical qualification. |
| [vLLM Jev](https://github.com/mode-io/vllm-jev) | Public serving project, Apache-2.0 as documented | Separates decision request serving from token-generation APIs; useful compatibility reference | Its vLLM execution is outside dllm's independent-runtime requirement. No package dependency added. |

NInfer's model documentation describes a draft-only vocabulary shortlist with
full-vocabulary target verification. A shortlist must never silently truncate
the target distribution in ordinary generation. Its GDN ReplaySSM documents
replaying raw accepted transitions from a checkpoint with the *same finite
precision operations* as verification; mathematical equivalence alone does not
prove recurrent-state equivalence. A future speculative path needs rejection,
rollback, EOS and stochastic-sampling proofs, plus actual end-to-end gains.

Open-Jev's inspected `jev/model.py` uses trained decision components; its
`third_party/open_jev_fast/src/fastmodel.py` packs shared context, question nodes
and candidate leaves and optionally fuses the final normalization/head. Branches
must not attend to one another. dllm must separately qualify any such tree masks,
recurrent branching and trained weight mappings before claiming support.

The first independent implementation in #33 scores an explicit bounded set of
next-token IDs after a single prompt prefill. It extracts only those output-head
rows, avoids generation, and returns probabilities normalized over that set.
This is a useful primitive for applications that explicitly choose token
verbalizers. It is not multi-token answer scoring, a Jev-compatible checkpoint
loader, a reranker or a calibrated confidence estimator. Hardware evidence is
kept separately in `docs/evidence/scoring/` once qualified.

References at the inspected NInfer pin:

- [Model and proposal head](https://github.com/Neroued/ninfer/blob/c772812b7f6f0cf893663b19971f49b6d9ce0e6e/docs/maintainer/qwen3_5-model.md)
- [Finite-precision recurrent replay](https://github.com/Neroued/ninfer/blob/c772812b7f6f0cf893663b19971f49b6d9ce0e6e/docs/maintainer/replayssm-gdn.md)
- [Resource scheduling and context cache](https://github.com/Neroued/ninfer/blob/c772812b7f6f0cf893663b19971f49b6d9ce0e6e/docs/maintainer/resource-scheduling-and-context-cache.md)

Subsequent independent work now includes trained BERT and multilingual
XLM-RoBERTa rerankers (#35/#36), distinct from proprietary Jev decision models,
and [full-target greedy n-gram speculation](evidence/speculation/) (#37).
The latter verifies complete-vocabulary target logits, preserves only the
accepted cache prefix, and qualifies rejection/EOS/slot/lifecycle behavior on
A2. Its proposal source is repeated token sequences, not NInfer's trained
MTP/DFlash components, and no recurrent or stochastic acceptance claim is made.
