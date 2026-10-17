"""Explicit CPU reference placement; never probes or admits an accelerator."""
from fractions import Fraction
from .cpu_hybrid_stage import cpu_stage_budget,CpuStageCapacityError,_layout


def plan_cpu_stages(model,budgets,*,max_batch=4,**stage_options):
    """Minimize maximum estimated utilization across ordered CPU byte budgets.

    Caller-supplied budgets must already reserve source mappings and unrelated
    process memory. Stage costs include conservative output/workspace ceilings.
    This plans storage only; it does not predict speed or transfer costs.
    """
    count=model.config.get('num_hidden_layers')
    if (type(count) is not int or not 1<=count<=256
            or not isinstance(budgets,(tuple,list)) or not 1<=len(budgets)<=min(16,count)
            or any(type(n) is not int or not 1<=n<=2**40 for n in budgets)):
        raise ValueError('CPU placement requires ordered positive byte budgets and nonempty bounded stages')
    _layout(model,0,count)  # Invalid storage is an error, never disguised as no-fit.
    candidates={}
    states={0:(Fraction(0),())}
    for rank,limit in enumerate(budgets):
        following={}
        for start,(score,ranges) in states.items():
            for end in range(start+1,count-(len(budgets)-rank-1)+1):
                key=(start,end)
                if key not in candidates:
                    try:candidates[key]=cpu_stage_budget(model,start,end,max_batch,**stage_options)
                    except CpuStageCapacityError:candidates[key]=None
                candidate=candidates[key]
                if candidate is None or candidate['required_memory_bytes']>limit:continue
                choice=(max(score,Fraction(candidate['required_memory_bytes'],limit)),(*ranges,key))
                if end not in following or choice<following[end]:following[end]=choice
        states=following
    if count not in states:
        return dict(storage_plan_found=False,cpu_budget_bytes=list(budgets),
                    reason='No contiguous CPU partition fits supplied storage and stage allocation limits')
    score,ranges=states[count]
    return dict(storage_plan_found=True,splits=[end for _,end in ranges[:-1]],maximum_estimated_utilization=float(score),
                cpu_budget_bytes=list(budgets),stages=[dict(candidates[key],cpu_budget_bytes=limit,
                    remaining_budget_bytes=limit-candidates[key]['required_memory_bytes']) for key,limit in zip(ranges,budgets)])
