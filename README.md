# Structure-Preserving Style Transfer using Canny, Depth & Flux

## Overview

This project implements a custom **image-to-image style transfer pipeline** that blends the **style of one image (Image A)** into the **structure of another image (Image B)**. 

We just added canny to  [this work by Nathan Shipley](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68), where the fusion of style and structure creates artistic visual outputs. It's an easy edut 

We will release the codes of the version leveraging Monolool architecture.

---

##  Key Features

-  **Style-Structure Fusion**: Seamlessly transfers style from Image A into the spatial geometry of Image B.
-  **Model-Driven Pipeline**: No UI dependencies; powered entirely through locally executed Python scripts.
-  **Modular**: Easily plug in other models or replace components (ControlNet, encoders, etc.).

---

##  How It Works

1. **Inputs**:
   - **Image A**: Style reference
   - **Image B**: Structural reference

2. **Structural Conditioning**:
   - **Canny Edge Map** of Image B
   - **Depth Map** via a pre-trained DepthAnything model

3. **Style Conditioning**:
   - Style prompts or embeddings extracted from Image A via a CLIP/T5/BLIP2 encoder

4. **Generation Phase**:
   - A diffusion model (e.g., Flux + Canny) is used
   - Flux-style injection merges the style and structure via guided conditioning
   - Output image retains Image B’s layout but adopts Image A’s artistic features


---

##  Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run generation**

```bash
gradio app.py
```



---

##  Use Cases

- AI-powered visual storytelling
- Concept art and virtual scene design
- Artistic remapping of real-world photos
- Ad creative generation

---

##  Credits & Inspiration

- [Nathan Shipley's work](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68) for the idea spark
- Hugging Face models:
  - [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)
  - [Flux Canny](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev)
  - [Flux dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev)
  - [DepthAnything](https://huggingface.co/Kijai/DepthAnythingV2-safetensors)

---



##  Contact

Want to collaborate or learn more? Reach out via GitHub or drop us a message!