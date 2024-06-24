import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

def initialise_pipeline(model = "runwayml/stable-diffusion-v1-5",
                        variant = "fp16",
                        dtype = torch.bfloat16):

    

    return pipeline, generator


def gen_image(pipeline, generator, prompt, negative_prompt, image, steps, strength = 0.7, guidance = 10):

    generated_image = pipeline(prompt,
        negative_prompt = negative_prompt,
        generator = generator,
        image=image,
        strength = strength,
        guidance_scale = guidance
        ).images[0]
   
    return generated_image
