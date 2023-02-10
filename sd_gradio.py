# pip install git+https://github.com/huggingface/optimum-graphcore.git
# pip install gradio pillow
# export PYTHONPATH=$PYTHONPATH:optimum-graphcore/notebooks/stable_diffusion/

import torch

from ipu_models import IPUStableDiffusionInpaintPipeline
import gradio as gr
from PIL import Image

executable_cache_dir = './exe_cache'
image_dimensions=(500,500)


import torch
from diffusers import DPMSolverMultistepScheduler

from ipu_models import IPUStableDiffusionPipeline



pipe = IPUStableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    revision="fp16",
    torch_dtype=torch.float16,
    ipu_config={
        "matmul_proportion": [0.06, 0.1, 0.1, 0.1],
        "executable_cache_dir": executable_cache_dir,
    },
    requires_safety_checker=False
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
image_width = 768
image_height = 768

pipe("apple", height=image_height, width=image_width, num_inference_steps=25, guidance_scale=9);

output = None

def generate(prompt, negative_prompt):
    # print(inputs['image'], inputs['mask'])
    global output
    output = pipe(prompt,
                  height=image_height,
                  width=image_width,
                  num_inference_steps=25,
                  guidance_scale=9).images[0]

    return output

if __name__ == '__main__':
    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.inputs.Textbox(lines=1, label='Prompt'),
            gr.inputs.Textbox(lines=1, label='Negative prompt'),
        ],
        outputs=[
            gr.Image(type="pil", )
        ],
        allow_flagging='never',
        title='Stable Diffusion in-painting')

    demo.launch(share=True)