# 🍡 Tiny AutoEncoder for Mochi 1

## What is TAEM1?

TAEM1 is a Tiny AutoEncoder for the Mochi 1 (Preview) video generation model.
TAEM1 should be able to encode/decode Mochi's latents more cheaply than the full-size Mochi VAE (at the cost of slightly lower quality).
This means TAEM1 should be useful for previewing outputs from Mochi 1.

| Sample Video | Reconstruction with TAEM1 |
| ------------ | ------------------------- |
| ![](https://github.com/user-attachments/assets/97560b3b-ea32-4ba8-9ca6-4c6be66b6976) | ![](https://github.com/user-attachments/assets/003aa013-b795-4c80-8a13-a1dd1f1b0b8b) |

## How does TAEM1 work?

TAEM1 consists of an MSE-distilled encoder and an MSE+adversarial-distilled decoder, both trained to mimic the Mochi 1 VAE behavior.
TAEM1 has the `vae_latents_to_dit_latents` and `dit_latents_to_vae_latents` transforms baked in, and consumes/produces [0, 1]-scaled images, so TAEM1 shouldn't require much additional code to use.
TAEM1 is causal (like the Mochi 1 VAE) so you can either run them timestep-parallel (faster, higher memory usage) or timestep-sequential (slower, reduced memory usage).

## How can I use TAEM1?

You can try running `python3 taem1.py test_video.mp4` to test reconstruction. Mochi T2V demo notebook TBD.
