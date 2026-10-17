"""Bound temporary prefill work without changing request order or cache slots."""


def groups(prompts, slots, token_budget):
    """Inputs are validated by the executor before any cache mutation."""
    batch, assigned, count = [], [], 0
    for prompt, slot in zip(prompts, slots):
        if batch and count + len(prompt) > token_budget:
            yield batch, assigned
            batch, assigned, count = [], [], 0
        batch.append(prompt)
        assigned.append(slot)
        count += len(prompt)
    if batch:
        yield batch, assigned
