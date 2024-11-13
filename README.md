# 🍡 Tiny AutoEncoder for Mochi 1

## What is TAEM1?

TAEM1 is a Tiny AutoEncoder for the Mochi 1 (Preview) video generation model.
TAEM1 should be able to encode/decode Mochi's latents more cheaply than the full-size Mochi VAE (at the cost of slightly lower quality).
This means TAEM1 should be useful for previewing outputs from Mochi 1.

| Sample Video | Reconstruction with TAEM1 |
| ------------ | ------------------------- |
| ![test_video_0 mp4](https://github.com/user-attachments/assets/5eabed4c-c942-43d6-9e6d-aeb2bba9fad0) | ![test_video_0 mp4 reconstructed_by_taem1 mp4](https://github.com/user-attachments/assets/4b77a175-8374-4b3d-a18a-f296aa696ab3) |

## How does TAEM1 work?

TAEM1 consists of an MSE-distilled encoder and an MSE+adversarial-distilled decoder, both trained to mimic the Mochi 1 VAE behavior.
TAEM1 has the `vae_latents_to_dit_latents` and `dit_latents_to_vae_latents` transforms baked in, and consumes/produces [0, 1]-scaled images, so TAEM1 shouldn't require much additional code to use.
TAEM1 is causal (like the Mochi 1 VAE) so you can either run TAEM1 timestep-parallel (faster, higher memory usage) or timestep-sequential (slower, reduced memory usage).

## How can I use TAEM1?

You can try running `python3 taem1.py test_video.mp4` to test reconstruction. Mochi T2V demo notebook TBD.
