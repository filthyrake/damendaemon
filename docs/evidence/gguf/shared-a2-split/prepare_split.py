from pathlib import Path
import gguf
source=Path('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q8_0.gguf')
reader=gguf.GGUFReader(str(source))
writer=gguf.GGUFWriter('/tmp/damendaemon-218/model.gguf', reader.fields['general.architecture'].contents(), split_max_tensors=150)
for key,field in reader.fields.items():
    if key.startswith('GGUF.') or key=='general.architecture':continue
    writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if len(field.types)>1 else None)
for tensor in reader.tensors:
    writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
from scripts.experiments.split_gguf import prepare
parts=sorted(Path('/tmp/damendaemon-218').glob('model-*.gguf'))
manifest=prepare(source,parts[0],Path('/tmp/damendaemon-218/evidence'))
print({'shards':len(parts),'tensor_count':len(manifest['sources']['single']['semantics']['tensors']),'prompt_ids':manifest['prompt_ids']})
