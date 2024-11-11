#!/usr/bin/env python3
"""
Tiny AutoEncoder for Mochi 1
(DNN for encoding / decoding videos to Mochi 1's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3
        
class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEM1 operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N*T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T]
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out, in_mem = [], None
        # iterate over input timesteps and also iterate over blocks
        # TODO(oboerbohan): this could be optimized to be more friendly to pytorch memory management
        for xt in tqdm(x.reshape(N, T * C, H, W).chunk(T, dim=1), disable=not show_progress_bar):
            out_mem = []
            for b in model:
                if isinstance(b, MemBlock):
                    out_mem.append(xt)
                    xt = b(xt, xt * 0 if in_mem is None else in_mem.pop(0))
                elif isinstance(b, TPool):
                    # needs special handling here since pools may use more memory
                    # length than the rest of the mem blocks do
                    if in_mem is None:
                        pool_mem = torch.cat([xt*0]*b.stride, 1)
                    else:
                        pool_mem = torch.cat([in_mem.pop(0), xt], 1)[:, -(b.stride*xt.shape[1]):]
                    out_mem.append(pool_mem)
                    _NT, C, H, W = xt.shape
                    xt = b(pool_mem.view(-1, C, H, W))
                else:
                    xt = b(xt)
            out.append(xt)
            in_mem = out_mem
        # TODO(oboerbohan): I think there's still a memory leak here or sth?
        x = torch.stack(out, 1)
    return x
        
class TAEM1(nn.Module):
    latent_channels = 12
    image_channels = 3
    def __init__(self, checkpoint_path="taem1.pth"):
        """Initialize pretrained TAEM1 from the given checkpoints."""
        super().__init__()
        self.encoder = nn.Sequential(
            conv(TAEM1.image_channels, 64), nn.ReLU(inplace=True),
            TPool(64, 3), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, TAEM1.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(TAEM1.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2), TGrow(n_f[1], 2), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2), TGrow(n_f[2], 3), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], TAEM1.image_channels),
        )
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=12) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        # NOTE:
        # the Mochi VAE does not preserve shape along the time axis;
        # videos are encoded to floor((n_in - 1)/6)+1 latent frames 
        # (which makes sense, it's stride 6, so 12 -> 2 and 13->3)
        # but then they're decoded to only the *minimal* number
        # of input frames (3 latents get decoded to 13 frames, not 18)
        # in order to achieve the intended causal structure...
        # anyway, that's why we have to remove some frames here.
        # mochi-VAE does the slicing at each TGrow (save compute/mem?)
        # but I think it's basically the same
        return x[:, 5:]

    def forward(self, x):
        return self.c(x)

@torch.no_grad()
def main():
    """Run TAEM1 roundtrip reconstruction on the given video paths."""
    import sys
    import cv2 # no highly esteemed deed is commemorated here

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        def __iter__(self):
            return self
        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration  # End of video or error
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
        def __del__(self):
            if hasattr(self, 'writer'): self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print("Using device", dev, "and dtype", dtype)
    taem1 = TAEM1().to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        # convert to device tensor
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            # convert to device tensor
            vid_enc = taem1.encode_video(vid_dev)
            print(f"  Encoded {video_path}. Decoding...")
            vid_dec = taem1.decode_video(vid_enc)
            print(f"  Decoded {video_path}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            # convert to device tensor
            vid_enc = taem1.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path}. Decoding...")
            vid_dec = taem1.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path}")
        video_out_path = video_path + ".reconstructed_by_taem1.mp4"
        video_out = VideoTensorWriter(video_out_path, (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        print(f"  Saved to {video_out_path}")

if __name__ == "__main__":
    main()
