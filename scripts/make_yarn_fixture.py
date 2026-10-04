"""Validation-only YaRN configuration fixture; preserves checkpoint weights.

Usage: python scripts/make_yarn_fixture.py SOURCE NEW_DIRECTORY
The shortened original context tests scaling boundaries economically; this is
not a pretrained long-context checkpoint or evidence of long-context quality.
"""
import json
import sys
from pathlib import Path
source=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2])
destination.mkdir(parents=True,exist_ok=False)
for item in source.iterdir():
    if item.name!='config.json':
        (destination/item.name).symlink_to(item)
config=json.loads((source/'config.json').read_text())
config['max_position_embeddings']=2048
config['rope_scaling']={'rope_type':'yarn','factor':4.,'original_max_position_embeddings':512}
(destination/'config.json').write_text(json.dumps(config,indent=2))
