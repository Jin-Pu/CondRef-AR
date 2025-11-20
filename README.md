
# CondRef-AR: Condition-as-a-Reference Randomized Autoregressive Modelling for Controllable Aerial Image Generation
[![Home](https://img.shields.io/badge/CondRef-AR?style=flat&label=Home
)](https://jin-pu.github.io/CondRef-AR)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://jin-pu.github.io/CondRef-AR) 
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/PuTorch/CondRef-AR)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Jin-Pu/CondRef-AR)

This repository contains the code and pretrained models for **CondRef-AR**, a controllable aerial image generation model using condition-as-a-reference randomized autoregressive modeling. The model generates high-quality aerial images based on input conditions such as sketches or segmentation maps.

![CondRef-AR Overview](assets/method.jpg)


## Quickstart

```python
import json, torch
from CondRefAR.pipeline import CondRefARPipeline
from transformers import AutoTokenizer, T5EncoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

gpt_cfg = json.load(open("configs/gpt_config.json"))
vq_cfg  = json.load(open("configs/vq_config.json"))
pipe = CondRefARPipeline.from_pretrained(".", gpt_cfg, vq_cfg, device=device, torch_dtype=dtype)

tok = AutoTokenizer.from_pretrained("google/flan-t5-xl")
enc = T5EncoderModel.from_pretrained("google/flan-t5-xl", torch_dtype=dtype).to(device).eval()

prompt = "Aaerial view of a forested area with a river running through it. On the right side of the image, there is a small town or village with a red-roofed building."
control = "assets/examples/example2.jpg"

from PIL import Image, ImageOps
control_img = Image.open(control).convert("RGB")

inputs = tok([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=120)
with torch.no_grad():
    emb = enc(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device)).last_hidden_state

imgs = pipe(emb, control_img, cfg_scale=4, temperature=1.0, top_k=2000, top_p=1.0)
imgs[0].save("sample.png")
```

## Sample Results
By varying the input conditions and prompts, CondRef-AR can generate diverse aerial images:
![Samples](assets/samples.png)

ConRef-AR can generate continuous, plausible, and high-resolution sequences of land-use change images based on a series of temporal semantic condition graphs. As shown in the figure below, the model successfully simulates the entire process—from a pristine forest gradually transforming into a modern residential urban area:

![Temporal Generation](assets/evolution.png)
<div align="center">

| Control image | Aerial image |
|---|---|
| <img src="assets/control_img.gif" alt="control animation" width="100%"/> | <img src="assets/aerial_img.gif" alt="aerial animation" width="100%"/> |

</div>
Please visit [Huggingface](https://huggingface.co/PuTorch/CondRef-AR) to download all weights 
## Files
- `weights/sketch-gpt-xl.safetensors`, `weights/vq-16.safetensors`: pretrained weight
- `configs/*.json`: model hyperparameters.
- `CondRefAR/*`: inference code and pipeline.
- `assets/example`: example images.
- `app.py`: Gradio demo.

## Notes
- Requires a GPU with bfloat16 support for best speed; CPU works but slow.
- CFG params: `cfg_scale`, `temperature`, `top_k`, `top_p` control quality vs diversity.
- If you have any questions, please open an issue, or contact putorch@outlook.com.

## License
Apache-2.0
