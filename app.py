import os
import time 
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

pipe.enable_attention_slicing()

history = []

def generate(prompt, negative_prompt, steps, guidance, width, height, seed):
    if prompt.strip() == "":
        return None, history
    
    if seed == -1:
        seed = torch.seed()

    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    ).images[0]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_seed{seed}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    
    history.insert(0, (path, prompt))
    
    return image, history

with gr.Blocks(title="AI Image Generator") as demo:
    
    gr.Markdown("# AI Image Generator")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Prompt Settings")
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter prompt here...",
                lines=3
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter negative prompt here...",
                lines=2
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 = random)"
                )

                steps = gr.Slider(
                    20, 40, value=28, step=1, label="Steps"
                )

                guidance = gr.Slider(
                    5, 12, value=8.5, step=0.5, label="Guidance Scale (CFG)"
                )

                width = gr.Dropdown(
                    [512, 768], value=512, label="Width"
                )

                height = gr.Dropdown(
                    [512, 768], value=512, label="Height"
                )

            btn = gr.Button("Generate", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Result")
            output = gr.Image(label="Generated Image", height=512, show_label=False)
    
    gr.Markdown("## History")
    
    gallery = gr.Gallery(
        label="Generated Images History",
        columns=4,
        height=400,
        show_label=False
    )
    
    btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output, gallery]
    )

demo.launch()
