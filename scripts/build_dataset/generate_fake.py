
import torch
import uuid
import os
from pathlib import Path
from diffusers import DiffusionPipeline

device = "cuda"
output_dir = Path("data/custom/fake")
output_dir.mkdir(parents=True, exist_ok=True)

SUFFIX = ", photorealistic, high quality, detailed, 4k, sharp focus, natural lighting"

PROMPTS = {
    "personas_famosas": [
        f"a photo of a famous politician giving a speech at a press conference{SUFFIX}",
        f"a celebrity walking the red carpet at an awards ceremony{SUFFIX}",
        f"world leaders shaking hands at an official summit{SUFFIX}",
        f"a famous musician performing live on stage in a concert{SUFFIX}",
        f"an athlete competing in a professional sports event{SUFFIX}",
    ],
    "eventos_sociales": [
        f"a large protest demonstration with crowds holding signs{SUFFIX}",
        f"aerial view of a flooded city after a natural disaster{SUFFIX}",
        f"a political rally with thousands of people gathered outdoors{SUFFIX}",
        f"a colorful street festival celebration with people dancing{SUFFIX}",
        f"a humanitarian aid distribution in a refugee camp{SUFFIX}",
    ],
    "paisajes_lugares": [
        f"a dramatic mountain landscape with snow peaks and pine forest{SUFFIX}",
        f"a busy urban cityscape at night with skyscrapers{SUFFIX}",
        f"a tropical ocean beach at golden sunset{SUFFIX}",
        f"a lush forest with a waterfall and a river{SUFFIX}",
        f"an aerial view of sand dunes in the Sahara desert{SUFFIX}",
    ],
    "animales": [
        f"a lion hunting in the savannah, wildlife photography{SUFFIX}",
        f"a golden retriever and a tabby cat playing together{SUFFIX}",
        f"a flock of birds in flight over a lake at dawn{SUFFIX}",
        f"colorful tropical fish swimming in a coral reef{SUFFIX}",
        f"cows and sheep grazing on a green countryside farm{SUFFIX}",
    ],
    "comida": [
        f"an elegantly plated gourmet meal in a fine dining restaurant{SUFFIX}",
        f"a vibrant farmers market stall full of fresh fruit{SUFFIX}",
        f"a street food vendor cooking in a busy Asian night market{SUFFIX}",
        f"freshly baked artisan bread and pastries in a bakery{SUFFIX}",
        f"a traditional Spanish paella dish served in a pan{SUFFIX}",
    ],
    "objetos_cotidianos": [
        f"a tidy home office desk with laptop books and coffee{SUFFIX}",
        f"a workbench full of hand tools and hardware{SUFFIX}",
        f"a flat lay of modern electronics gadgets and smartphones{SUFFIX}",
        f"a collection of fashionable clothing and accessories{SUFFIX}",
        f"an open book with stationery items on a wooden desk{SUFFIX}",
    ],
}

IMAGES_PER_PROMPT = 30

print("Cargando SDXL...")
torch.cuda.empty_cache()
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
print("Modelo cargado ✓")

total = 0
for category, prompts in PROMPTS.items():
    cat_dir = output_dir / category
    cat_dir.mkdir(exist_ok=True)
    print(f"\n━━━ {category} ━━━")

    for p_idx, prompt in enumerate(prompts, 1):
        print(f"  [{p_idx}/{len(prompts)}] {prompt[:60]}...")

        for i in range(IMAGES_PER_PROMPT):
            image = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

            filename = cat_dir / f"{uuid.uuid4().hex}.png"
            image.save(str(filename))
            total += 1

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{IMAGES_PER_PROMPT} ({total} total)")

print(f"\n✓ Completado: {total} imágenes en {output_dir}")
