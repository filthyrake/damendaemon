"""Explicit operator-configured text chat for the bounded CPU reference path."""
from contextlib import closing
from io import StringIO
import hashlib
import json
from pathlib import Path
from .chat_template import compile_template,read_override,RENDER_ERRORS
from .cpu_reference_text import CpuReferenceText,read_json


class CpuReferenceChat(CpuReferenceText):
    def __init__(self,path,config,template_file):
        if config.get('model_type')!='qwen3_5_text':raise ValueError('CPU chat requires text Qwen3.5')
        template=read_override(template_file)
        super().__init__(path,config)
        metadata_path=Path(path)/'tokenizer_config.json'
        metadata=read_json(metadata_path,1024**2) if metadata_path.exists() else {}
        if not isinstance(metadata,dict):raise ValueError('invalid CPU chat tokenizer metadata')
        special={}
        for name in ('bos_token','eos_token'):
            token=metadata.get(name)
            if isinstance(token,dict):
                if not isinstance(token.get('content'),str):raise ValueError('invalid CPU chat special token metadata')
                token=token['content']
            if token is not None and (not isinstance(token,str) or not token or len(token.encode())>4096
                                      or self.tokenizer.token_to_id(token) is None):
                raise ValueError('CPU chat special tokens must be mapped tokenizer strings')
            special[name]=token
        self.template=compile_template(template);self.special_tokens=special
        self.chat_identity=hashlib.sha256(json.dumps(dict(template=template,special_tokens=special),
            sort_keys=True,separators=(',',':')).encode()).digest()

    def chat_tokens(self,messages,*,enable_thinking=None):
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError('CPU chat thinking option must be boolean')
        if not isinstance(messages,list) or not 1<=len(messages)<=256:
            raise ValueError('CPU chat requires 1..256 text messages')
        total=0;prepared=[]
        for message in messages:
            if (not isinstance(message,dict) or set(message)!={'role','content'}
                    or message['role'] not in ('system','user','assistant') or not isinstance(message['content'],str)):
                raise ValueError('CPU chat supports only text system/user/assistant messages')
            total+=len(message['content'].encode('utf-8'))
            if total>1024**2:raise ValueError('CPU chat messages exceed 1 MiB')
            prepared.append(dict(message))
        options={} if enable_thinking is None else dict(enable_thinking=enable_thinking)
        total=0
        try:
            with StringIO() as output, closing(self.template.generate(messages=prepared,
                    add_generation_prompt=True,tools=None,**options,**self.special_tokens)) as stream:
                for chunk in stream:
                    total+=len(chunk.encode('utf-8'))
                    if total>1024**2:raise ValueError('CPU chat rendered prompt exceeds 1 MiB')
                    output.write(chunk)
                rendered=output.getvalue()
        except RENDER_ERRORS as error:raise ValueError('CPU chat template rendering failed') from error
        return self.tokenizer.encode(rendered,add_special_tokens=False).ids
