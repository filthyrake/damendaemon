"""Independent xgrammar acceptance tests for correlated function/argument unions."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import xgrammar as xgr
from transformers import AutoTokenizer
from dllm.tool_calls import prepare_tools
from verify_forced_tools import TOOLS


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    info=xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler=xgr.GrammarCompiler(info,max_threads=2)
    cases=[]
    for choice in ('required',{'type':'function','function':{'name':'translate'}}):
        _,schema=prepare_tools(TOOLS,choice)
        compiled=compiler.compile_json_schema(schema,strict_mode=True,max_whitespace_cnt=8)
        for value in [
            {'name':'weather','arguments':{'city':'Paris'}},
            {'name':'translate','arguments':{'phrase':'bonjour'}},
            {'name':'weather','arguments':{'phrase':'bonjour'}},
            {'name':'translate','arguments':{'city':'Paris'}},
            {'name':'unknown','arguments':{}},
            {'name':'weather','arguments':{'city':'London'}},
            {'name':'weather','arguments':{}},
            {'name':'weather','arguments':{'city':'Paris','extra':True}},
        ]:
            expected=value=={'name':'translate','arguments':{'phrase':'bonjour'}} or (
                choice=='required' and value=={'name':'weather','arguments':{'city':'Paris'}})
            matcher=xgr.GrammarMatcher(compiled)
            accepted=matcher.accept_string(json.dumps(value)) and matcher.is_completed()
            assert accepted==expected,(choice,value,accepted,expected)
            cases.append({'choice':choice,'value':value,'accepted':accepted})
    print(json.dumps({'passed':True,'cases':cases},indent=2))


if __name__=='__main__':main()
