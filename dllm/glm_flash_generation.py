"""Ordinary Flash CPU generation using the shared owned GLM token lifecycle."""
from .glm_generation import ReferenceGenerator
from .glm_flash_model import FlashModel
from .glm_flash_text import FlashText


def flash_text_config(config):
    if (not isinstance(config,dict) or config.get('model_type')!='glm5_next'
            or not isinstance(config.get('text_config'),dict)
            or config['text_config'].get('model_type')!='glm5_next_text'):
        raise ValueError('Flash generation requires a Flash root/text configuration')
    return config['text_config']


class FlashReferenceGenerator(ReferenceGenerator):
    model_type=FlashModel
    text_type=FlashText

    def _configuration(self,model):
        config=getattr(model,'config',None)
        return {} if config is None else flash_text_config(config)
