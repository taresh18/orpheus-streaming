from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import os
import time
import logging

logger = logging.getLogger(__name__)

torch.set_num_threads(1)

# Initialize SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(snac_device)
if snac_device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Warm up the model with a dummy inference
    dummy_codes = [
        torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=snac_device),
        torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=snac_device),
        torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=snac_device)
    ]
    with torch.inference_mode():
        _ = model.decode(dummy_codes)

# Load decoder configuration from environment variables
CUSTOM_TOKEN_PREFIX = "<custom_token_"
CUSTOM_TOKEN_SUFFIX = ">"
MIN_FRAMES_FIRST = int(os.environ["MIN_FRAMES_FIRST"])
MIN_FRAMES_SUBSEQ = int(os.environ["MIN_FRAMES_SUBSEQ"])
PROCESS_EVERY = int(os.environ["PROCESS_EVERY"])

def turn_token_into_id(token_string, index):
    """Convert a custom token string to its numeric ID.

    Args:
        token_string (str): The literal token text coming from the model.
        index (int): Absolute token position (used for offset calculation).

    Returns:
        Optional[int]: Numeric token ID or ``None`` if the token is invalid.
    """
    token_string = token_string.strip()
    mod = index % 7

    try:
        digits_str = token_string.removeprefix(CUSTOM_TOKEN_PREFIX).removesuffix(CUSTOM_TOKEN_SUFFIX)
        token_id = int(digits_str) - 10 - (mod * 4096)
        return token_id
    except (ValueError, TypeError):
        return None


def convert_to_audio(multiframe, count):
    """
    Highly optimized version of convert_to_audio that eliminates inefficient 
    tensor operations and reduces CPU-GPU transfers for much faster inference
    on high-end GPUs. Optimized for concurrent requests.
    """
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    
    # Pre-allocate tensors with the right shape and directly on target device
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Fill tensors with direct indexing
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    # validation for range check
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        return None
    
    codes = [codes_0, codes_1, codes_2]
    
    with torch.inference_mode():   
        audio_hat = model.decode(codes)
        audio_slice = audio_hat[:, :, 2048:4096]
        
        if snac_device == "cuda":
            audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
            return audio_int16_tensor.cpu().numpy().tobytes()
        else:
            audio_np = audio_slice.numpy()
            return (audio_np * 32767.0).round().astype(np.int16).tobytes()


async def tokens_decoder(token_gen):
    """Decode tokens into audio chunks with reduced latency.

    The first audio chunk is emitted as soon as **one** frame (7 tokens) is
    available, drastically reducing time-to-first-byte. Subsequent chunks are
    processed every 7 tokens using a sliding window of the last 4 frames (28
    tokens) mirroring the original behaviour.
    """    
    buffer = []
    count = 0
    first_chunk_sent = False

    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None or token <= 0:
            continue

        buffer.append(token)
        count += 1

        if not first_chunk_sent and count >= MIN_FRAMES_FIRST:
            audio = convert_to_audio(buffer[-MIN_FRAMES_FIRST:], count)
            if audio is not None:
                first_chunk_sent = True
                yield audio
        elif first_chunk_sent and count % PROCESS_EVERY == 0:
            audio = convert_to_audio(buffer[-MIN_FRAMES_SUBSEQ:], count)
            if audio is not None:
                yield audio