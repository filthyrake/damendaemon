"""Real GPU lease cancellation and weight-budget rejection/recovery checks."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json

from dllm.runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    args = parser.parse_args()
    # A=Qwen2.5-0.5B fits; B=Qwen2.5-1.5B exceeds this host weight budget.
    runtime = Runtime({"a": args.model_a, "b": args.model_b}, context=256, host_gib=1.25)
    question = [{"role": "user", "content": "What is 2 + 2? Answer with one number."}]
    first = list(runtime.generate("a", question, 32))
    text = "".join(event.get("text", "") for event in first)
    assert text.strip() == "4", text
    assert first[-1]["finish_reason"] == "stop", first[-1]
    try:
        list(runtime.generate("b", question, 32))
    except MemoryError:
        pass
    else:
        raise AssertionError("oversized model was admitted")
    assert list(runtime.gpu) == ["a"], "budget rejection evicted the usable model"
    pending = runtime.generate("a", [{"role": "user", "content": "Explain linear algebra in detail."}], 128)
    next(pending)
    pending.close()
    recovered = list(runtime.generate("a", question, 32))
    assert "".join(event.get("text", "") for event in recovered) == text
    assert recovered[-1]["timing"]["source_tier"] == "gpu"
    print(json.dumps({"eos_stopping": True, "weight_budget_rejection": True,
                      "retained_ready_model": True, "cancelled_generator_recovery": True,
                      "answer": text, "usage": recovered[-1]["usage"]}, indent=2))


if __name__ == "__main__":
    main()
