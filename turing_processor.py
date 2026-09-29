import subprocess
import json
import sys
import numpy as np
import torch
import kornia as K
from tqdm import tqdm
import threading
import queue

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True 

def get_video_metadata(input_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'json', input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)['streams'][0]
    num, den = map(int, info['r_frame_rate'].split('/'))
    return {
        'width': int(info['width']),
        'height': int(info['height']),
        'fps': num / den,
        'frames': int(info.get('nb_frames', 0))
    }

def get_1d_gaussian_kernel(kernel_size, sigma, device):
    k = K.filters.get_gaussian_kernel1d(kernel_size, sigma).to(device)
    return k.view(1, 1, -1, 1), k.view(1, 1, 1, -1)

@torch.compile
def turing_loop(x, k5_v, k5_h, k7_v, k7_h, iterations=100, sharpen_strength=5.0):
    for _ in range(iterations):
        # REACTION (Separable 5x5 blur)
        blur_small = torch.nn.functional.conv2d(x, k5_v, padding=(2, 0))
        blur_small = torch.nn.functional.conv2d(blur_small, k5_h, padding=(0, 2))
        
        details = x - blur_small
        x = x + (details * sharpen_strength)
        
        # DIFFUSION (Separable 7x7 blur)
        x = torch.nn.functional.conv2d(x, k7_v, padding=(3, 0))
        x = torch.nn.functional.conv2d(x, k7_h, padding=(0, 3))
        
        x = torch.clamp(x, 0.0, 1.0)
    return x

def reader_worker(process, batch_bytes, q_out):
    """Reads a massive chunk of raw bytes equal to the entire batch size at once."""
    while True:
        chunk = process.stdout.read(batch_bytes)
        if not chunk:
            q_out.put(None) 
            break
        q_out.put(chunk)

def writer_worker(process, q_in):
    """Writes massive chunks of processed bytes directly to FFmpeg."""
    while True:
        out_bytes = q_in.get()
        if out_bytes is None:
            break
        process.stdin.write(out_bytes)

def process_video_block_optimized(input_path, output_path, iterations=100, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing on: {device} | Batch Size: {batch_size}")
    
    meta = get_video_metadata(input_path)
    width, height = meta['width'], meta['height']
    
    # Calculate exact byte sizes for 1-channel grayscale
    frame_bytes = width * height
    batch_bytes = frame_bytes * batch_size

    k5_v, k5_h = get_1d_gaussian_kernel(5, 1.0, device)
    k7_v, k7_h = get_1d_gaussian_kernel(7, 1.8, device)

    read_cmd = [
        'ffmpeg', '-i', input_path, 
        '-f', 'image2pipe', '-pix_fmt', 'gray', '-vcodec', 'rawvideo', '-'
    ]
    read_process = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    write_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
        '-pix_fmt', 'gray', '-r', str(meta['fps']), '-i', '-', '-c:v', 'libx264', 
        '-preset', 'ultrafast',  # <--- CHANGED FROM 'fast'
        '-pix_fmt', 'yuv420p', output_path
    ]


    write_process = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Queue maxsize is small because each item is now a massive block of memory
    read_queue = queue.Queue(maxsize=3)
    write_queue = queue.Queue(maxsize=3)

    reader_thread = threading.Thread(target=reader_worker, args=(read_process, batch_bytes, read_queue))
    writer_thread = threading.Thread(target=writer_worker, args=(write_process, write_queue))
    reader_thread.start()
    writer_thread.start()

    pbar = tqdm(total=meta['frames'], desc="Processing Frames")
    
    while True:
        chunk = read_queue.get()
        if chunk is None:
            break
            
        # Handle the final batch which might be smaller than the full batch_size
        num_frames = len(chunk) // frame_bytes
        if num_frames == 0:
            break

        # ZERO-COPY RESHAPE: Instantly wrap the raw memory block as a NumPy array
        batch_np = np.frombuffer(chunk[:num_frames * frame_bytes], dtype=np.uint8).copy().reshape(num_frames, 1, height, width)
        # PUSH TO GPU: Pin memory for fast async transfer
        batch_t = torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True, dtype=torch.float32) / 255.0
        # GPU COMPUTE
        batch_t = turing_loop(batch_t, k5_v, k5_h, k7_v, k7_h, iterations=iterations)

        # ZERO-COPY WRITE: Pull to CPU and instantly cast back to byte buffer
        out_np = (batch_t.cpu().numpy() * 255).astype(np.uint8)
        write_queue.put(out_np.tobytes())
        
        pbar.update(num_frames)

    # Teardown
    write_queue.put(None)
    reader_thread.join()
    writer_thread.join()
    read_process.wait()
    write_process.wait()
    pbar.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python turing_processor.py <input_video> <output_video>")
        sys.exit(1)
        
    process_video_block_optimized(sys.argv[1], sys.argv[2], iterations=100, batch_size=128)
