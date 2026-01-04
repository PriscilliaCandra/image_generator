import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

def generate(prompt, negative_prompt, steps, guidance, width, height):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    ).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("Image Generator")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter prompt here..."
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter negative prompt here..."
            )
            
            steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
            guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale")

            width = gr.Dropdown([384, 512, 768], value=512, label="Width")
            height = gr.Dropdown([384, 512, 768], value=512, label="Height")

            btn = gr.Button("Generate")
            
        with gr.Column():
            output = gr.Image(label="Result")
            
    btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, steps, guidance, width, height],
        outputs=output
    )

demo.launch()
