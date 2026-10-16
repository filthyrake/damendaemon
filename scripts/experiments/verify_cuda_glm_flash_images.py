"""Guarded tiny jointly placed CUDA Flash image features and owned generation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_glm_flash_image_placement import place_cuda_flash_image
    from dllm.cuda_glm_flash_image_generation import CudaFlashImageToolGenerator,cuda_image_embeddings
    from dllm.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
    from dllm.glm_flash_image_tool_text import FlashImageToolText
    from dllm.glm_flash_image_embeddings import image_embeddings
    from dllm.glm_flash_images import prepare_images
    from tests.test_cuda_glm_flash_placement import fixture
    from tests.test_glm_flash_image_generation import text_fixture,vision
    from tests.test_glm_flash_vision import control
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,official_vision_qualified=False)
    save(args.output,report);states=[]
    try:
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);base=text_fixture(path);text=FlashImageToolText(path,vocab_size=base.vocab_size)
            source=fixture();source.embedding=torch.randn(text.vocab_size,4)*.1;source.head=torch.randn(text.vocab_size,4)*.1
            encoder=vision(4);placed=place_cuda_flash_image(source,encoder,device='cuda:0',max_bytes=1024**2)
            gpu=placed.model;images=[torch.full((112,112,3),127,dtype=torch.uint8)]
            messages=[dict(role='user',content=[dict(type='text',text='Hi'),dict(type='image')])]
            patches,grids=prepare_images(images,max_patches=4096)
            expected_features=control(patches,encoder.c,encoder.w,grids[0])
            features=placed.vision.forward(patches.cuda(),grids)
            torch.testing.assert_close(features.cpu(),expected_features,rtol=3e-5,atol=3e-6)
            tokens=torch.tensor([text.image_chat_tokens(messages,grids)])
            expected_values=image_embeddings(source,tokens,[expected_features],image_token_id=text.image_token_id)
            values=cuda_image_embeddings(gpu,tokens.cuda(),[features],image_token_id=text.image_token_id)
            torch.testing.assert_close(values.cpu(),expected_values,rtol=3e-5,atol=3e-6)
            expected,reference=source.prefill_embeddings(expected_values,capacity=tokens.shape[1]+2);states.append(reference)
            actual,state=gpu.prefill_embeddings(values,capacity=tokens.shape[1]+2);states.append(state)
            torch.testing.assert_close(actual.cpu(),expected,rtol=8e-5,atol=4e-6)
            torch.testing.assert_close(actual.argmax(-1).cpu(),expected.argmax(-1),rtol=0,atol=0)
            report.update(feature_max_absolute=float((features.cpu()-expected_features).abs().max()),logit_max_absolute=float((actual.cpu()-expected).abs().max()),placement_raw_bytes=placed.raw_bytes)
            for s in states:s.close()
            # Zero head keeps untrained controls out of the protocol comparison.
            source.head.zero_();gpu.head.zero_()
            cpu_runner=FlashImageToolReferenceGenerator(source,text,encoder,max_context=2048)
            runner=CudaFlashImageToolGenerator(gpu,text,placed.vision,max_context=2048)
            expected_events=list(cpu_runner.generate_images(messages,images,max_new_tokens=2))
            events=list(runner.generate_images(messages,images,max_new_tokens=2))
            assert events==expected_events
            original=gpu.prefill_embeddings
            def capture(*args,**kwargs):
                result=original(*args,**kwargs);states.append(result[1]);return result
            with patch.object(gpu,'prefill_embeddings',side_effect=capture):
                iterator=runner.generate_images(messages,images,max_new_tokens=3);next(iterator);iterator.close()
            assert states[-1].closed and not runner.failed_cleanups and not gpu.failed_cleanups
            report.update(passed=True,greedy_exact=True,image_generation_exact=True,abandonment_closed=True,
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for state in states:state.close()
        save(args.output,report)


if __name__=='__main__':main()
