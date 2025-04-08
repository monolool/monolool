# Structure-Preserving Style Transfer using Canny, Depth & Flux

## Overview

This project implements a custom **image-to-image style transfer pipeline** that blends the **style of one image (Image A)** into the **structure of another image (Image B)**. 

We just added canny to  [this work by Nathan Shipley](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68), where the fusion of style and structure creates artistic visual outputs. It's an easy edut 

We will release the codes of the version leveraging ZenCtrl architecture.

---

## �� Key Features

- �� **Style-Structure Fusion**: Seamlessly transfers style from Image A into the spatial geometry of Image B.
- �� **Model-Driven Pipeline**: No UI dependencies; powered entirely through locally executed Python scripts.
- �� **Modular**: Easily plug in other models or replace components (ControlNet, encoders, etc.).

---

## �� How It Works

1. **Inputs**:
   - **Image A**: Style reference
   - **Image B**: Structural reference

2. **Structural Conditioning**:
   - **Canny Edge Map** of Image B
   - **Depth Map** via a pre-trained MiDaS model or similar

3. **Style Conditioning**:
   - Style prompts or embeddings extracted from Image A via a CLIP/T5/BLIP2 encoder

4. **Generation Phase**:
   - A diffusion model (e.g., Stable Diffusion + ControlNet) is used
   - Flux-style injection merges the style and structure via guided conditioning
   - Output image retains Image B’s layout but adopts Image A’s artistic features

---

## �� Project Structure

```
├── models/
│   ├── controlnet/
│   ├── stable-diffusion/
├── scripts/
│   ├── extract_edges.py       # Canny edge map from Image B
│   ├── estimate_depth.py      # Generate depth map
│   ├── encode_style.py        # Encode Image A (prompt or features)
│   └── generate.py            # Full generation pipeline
├── assets/
│   ├── input_style.jpg
│   ├── input_structure.jpg
│   └── output.jpg
├── README.md
```

---

## �� Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run generation**

```bash
python scripts/generate.py \
  --style_image assets/input_style.jpg \
  --structure_image assets/input_structure.jpg \
  --output_image assets/output.jpg
```

Options:
- `--use_depth true`
- `--use_canny true`
- `--style_prompt "a painting in van Gogh style"`

---

## �� Example

| Style (Image A) | Structure (Image B) | Output |
|----------------|---------------------|--------|
| ![](assets/input_style.jpg) | ![](assets/input_structure.jpg) | ![](assets/output.jpg) |

---

## �� Use Cases

- AI-powered visual storytelling
- Concept art and virtual scene design
- Artistic remapping of real-world photos
- Ad creative generation

---

## �� Credits & Inspiration

- [Nathan Shipley's work](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68) for the idea spark
- Hugging Face models:
  - [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4)
  - [ControlNet for Canny/Depth](https://huggingface.co/lllyasviel/ControlNet)
  - [MiDaS for Depth](https://huggingface.co/Intel/dpt-large)
- Flux-based prompt handling inspired by multimodal conditioning techniques

---

---

## �� Try Also: ZenCtrl – Modular Agentic Image Control

If you enjoyed this project, you may also like [**ZenCtrl**](https://github.com/FotographerAI/ZenCtrl), our open-source **agentic visual control toolkit** for generative image pipelines that we are developing.

ZenCtrl can be combined with this style transfer project to introduce additional layers of control, allowing for more refined composition before or after stylization. It’s especially useful when working with structured scenes, human subjects, or product imagery.

With ZenCtrl, we aim to:
- Chain together preprocessing, control, editing, and postprocessing modules
- Create workflows for tasks like **product photography**, **try-on**, **background swaps**, and **face editing**
- Use control adapters like **canny, depth, pose, segmentation**, and more
- Easily integrate with APIs or run it in a Hugging Face Space

> Whether you're refining structure by changing the background layout before stylization or editing the results afterward, ZenCtrl gives you full compositional control across the image generation stack.

👉 [Explore ZenCtrl on GitHub](https://github.com/FotographerAI/ZenCtrl)
👉 [Try the ZenCtrl Demo on Hugging Face Spaces](https://huggingface.co/spaces/fotographerai/ZenCtrl)

---


## �� Contact

Want to collaborate or learn more? Reach out via GitHub or drop us a message!