import requests
import time
import statistics
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env file
SERVER_HOST = "localhost"
SERVER_PORT = "9090"
ENDPOINT_TYPE = os.getenv("BENCHMARK_ENDPOINT", "trtserve")  # "original" or "trtserve"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
TEST_TEXT = "Hi How are you doing today? So happy to see you again."
VOICE = "tara"
NUM_RUNS = 5
CONCURRENT_REQUESTS = 12
WARMUP_TEXT = "Doing warmup"
OUTPUT_DIR = "outputs"

# Audio parameters for WAV header generation
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1

# Create outputs directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_wav_header(sample_rate=SAMPLE_RATE, bits_per_sample=BITS_PER_SAMPLE, channels=CHANNELS, data_size=0):
    """Generate WAV header for PCM audio data."""
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    
    # Calculate file size (header + data)
    file_size = 36 + data_size
    
    # Build WAV header
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', file_size))
    header.extend(b'WAVE')
    # Format chunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Format chunk size
    header.extend(struct.pack('<H', 1))   # PCM format
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', byte_rate))
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    # Data chunk
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    
    return bytes(header)

def run_single_test(text, voice, save_file=None, session=None, request_id=None):
    """Run a single TTS test and return timing metrics.

    Parameters
    ----------
    text : str
        Prompt text to send to the TTS service.
    voice : str
        Voice name.
    save_file : str or None, optional
        If provided, the received raw PCM stream will be wrapped with a WAV header
        and written to this path.
    session : requests.Session or None, optional
        If supplied, the session's ``post`` method will be used, allowing the
        underlying TCP connection to be kept alive across multiple runs.  If
        ``None`` (default) a standalone ``requests.post`` call is made.
    request_id : int or None, optional
        Identifier for the request when running concurrently.
    """
    start_time = time.time()
    ttfb = None

    # Use the provided session for connection-pooled requests, or fall back to
    # the top-level ``requests`` module.
    http = session or requests

    try:
        response = http.post(
            f"{BASE_URL}/v1/audio/speech/stream",
            json={
                "input": text,
                "voice": voice,
            },
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            bytes_received = 0
            audio_data = bytearray()
            
            # Initialize an iterator for the response content with a more efficient chunk size.
            content_iterator = response.iter_content(chunk_size=4096)
            
            # Process chunks and capture TTFB when the first non-empty chunk arrives
            for chunk in content_iterator:
                if not chunk:
                    # Skip keep-alive chunks or empty payloads
                    continue
                if ttfb is None:
                    ttfb = time.time() - start_time  # First byte received
                audio_data.extend(chunk)
                bytes_received += len(chunk)
            
            # No audio was received – treat as failure
            if bytes_received == 0:
                return {'success': False, 'error': 'No audio data received (zero bytes)', 'request_id': request_id}
            
            # Save the complete audio as a WAV file if requested.
            if save_file:
                print(f" -> Saving {len(audio_data)} bytes to {save_file}", end="")
                wav_header = generate_wav_header(data_size=len(audio_data))
                with open(save_file, 'wb') as f:
                    f.write(wav_header)
                    f.write(audio_data)
            
            total_time = time.time() - start_time
            
            bytes_per_second = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
            audio_duration = len(audio_data) / bytes_per_second if bytes_per_second > 0 else 0

            return {
                'success': True,
                'ttfb': ttfb,
                'total_time': total_time,
                'bytes_received': bytes_received,
                'audio_duration': audio_duration,
                'request_id': request_id
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}", 'request_id': request_id}
            
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f"Request failed: {e}", 'request_id': request_id}
    except Exception as e:
        return {'success': False, 'error': f"An unexpected error occurred: {e}", 'request_id': request_id}

def run_concurrent_tests(text, voice, num_concurrent=CONCURRENT_REQUESTS, save_files=False):
    """Run multiple TTS tests concurrently and return timing metrics.
    
    Parameters
    ----------
    text : str
        Prompt text to send to the TTS service.
    voice : str
        Voice name.
    num_concurrent : int
        Number of concurrent requests to make.
    save_files : bool
        Whether to save audio files for each request.
    """
    results = []
    
    def run_request(request_id):
        save_file = None
        if save_files:
            save_file = f"{OUTPUT_DIR}/concurrent_run_{request_id}.wav"
        return run_single_test(text, voice, save_file, request_id=request_id)
    
    print(f"Starting {num_concurrent} concurrent requests...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # Submit all requests
        future_to_id = {executor.submit(run_request, i): i for i in range(num_concurrent)}
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    print(f"Request {request_id}: TTFB {result['ttfb']:.3f}s, Total {result['total_time']:.3f}s")
                else:
                    print(f"Request {request_id}: Failed - {result['error']}")
            except Exception as e:
                print(f"Request {request_id}: Exception - {e}")
                results.append({'success': False, 'error': str(e), 'request_id': request_id})
    
    total_concurrent_time = time.time() - start_time
    
    return results, total_concurrent_time

def main():
    print("TTS Streaming Benchmark")
    print(f"Endpoint Type: {ENDPOINT_TYPE}")
    print(f"Text: '{TEST_TEXT}' ({len(TEST_TEXT)} characters)")
    print(f"Voice: {VOICE}")
    print(f"Sequential Runs: {NUM_RUNS}")
    print(f"Concurrent Requests: {CONCURRENT_REQUESTS}")
    print(f"Base URL: {BASE_URL}")
    print("Note: Converting raw audio streams to WAV format with proper headers")
    
    # Create a single session so every run reuses the same HTTP connection.
    with requests.Session() as session:

        # Warmup run
        print("\nRunning warmup...")
        warmup_result = run_single_test(WARMUP_TEXT, VOICE, session=session)
        if warmup_result['success']:
            print(f"Warmup complete: TTFB {warmup_result['ttfb']:.3f}s, Total {warmup_result['total_time']:.3f}s")
        else:
            print(f"Warmup failed: {warmup_result['error']}")
        
        # Sequential benchmark runs
        print(f"\nStarting {NUM_RUNS} sequential benchmark runs...")
        sequential_results = []
        
        for run in range(1, NUM_RUNS + 1):
            print(f"Run {run}/{NUM_RUNS}...", end=" ")
            
            # Save as .wav files with proper headers
            audio_filename = f"{OUTPUT_DIR}/run_{run}.wav"
            result = run_single_test(TEST_TEXT, VOICE, audio_filename, session=session)
            
            if result['success']:
                print(f" TTFB: {result['ttfb']:.3f}s, Total: {result['total_time']:.3f}s")
                sequential_results.append(result)
            else:
                print(f" Failed: {result['error']}")
        
        # Concurrent benchmark runs
        print(f"\nStarting {NUM_RUNS} rounds of {CONCURRENT_REQUESTS} concurrent requests...")
        concurrent_results = []
        
        for round_num in range(1, NUM_RUNS + 1):
            print(f"\nRound {round_num}/{NUM_RUNS}:")
            round_results, round_time = run_concurrent_tests(TEST_TEXT, VOICE, CONCURRENT_REQUESTS, save_files=True)
            concurrent_results.extend(round_results)
            print(f"Round {round_num} completed in {round_time:.3f}s")
        
        # Calculate and display statistics
        print(f"\n--- Sequential Benchmark Results ---")
        # Initialize variables to avoid UnboundLocalError
        ttfb_times = []
        total_times = []
        audio_durations = []
        rtfs = []
        
        if sequential_results:
            successful_runs = len(sequential_results)
            ttfb_times = [r['ttfb'] for r in sequential_results if r['ttfb'] is not None]
            total_times = [r['total_time'] for r in sequential_results]
            audio_durations = [r['audio_duration'] for r in sequential_results if r.get('audio_duration')]
            rtfs = [r['total_time'] / r['audio_duration'] for r in sequential_results if r.get('audio_duration', 0) > 0]
            
            print(f"Successful runs: {successful_runs}/{NUM_RUNS}")
            
            if audio_durations:
                avg_duration = statistics.mean(audio_durations)
                print(f"Average Audio Duration: {avg_duration:.3f}s")
            
            if ttfb_times:
                ttfb_mean = statistics.mean(ttfb_times)
                ttfb_stdev = statistics.stdev(ttfb_times) if len(ttfb_times) > 1 else 0
                print(f"TTFB - Mean: {ttfb_mean:.3f}s, StdDev: {ttfb_stdev:.3f}s, Min: {min(ttfb_times):.3f}s, Max: {max(ttfb_times):.3f}s")
            
            if total_times:
                total_mean = statistics.mean(total_times)
                total_stdev = statistics.stdev(total_times) if len(total_times) > 1 else 0
                print(f"Total Time - Mean: {total_mean:.3f}s, StdDev: {total_stdev:.3f}s, Min: {min(total_times):.3f}s, Max: {max(total_times):.3f}s")

            if rtfs:
                rtf_mean = statistics.mean(rtfs)
                rtf_stdev = statistics.stdev(rtfs) if len(rtfs) > 1 else 0
                print(f"RTF - Mean: {rtf_mean:.3f}, StdDev: {rtf_stdev:.3f}, Min: {min(rtfs):.3f}, Max: {max(rtfs):.3f}")
        else:
            print("No successful sequential runs completed.")
        
        print(f"\n--- Concurrent Benchmark Results ---")
        if concurrent_results:
            successful_concurrent = len([r for r in concurrent_results if r['success']])
            total_concurrent = len(concurrent_results)
            concurrent_ttfb_times = [r['ttfb'] for r in concurrent_results if r['success'] and r['ttfb'] is not None]
            concurrent_total_times = [r['total_time'] for r in concurrent_results if r['success']]
            
            print(f"Successful concurrent requests: {successful_concurrent}/{total_concurrent}")
            
            if concurrent_ttfb_times:
                ttfb_mean = statistics.mean(concurrent_ttfb_times)
                ttfb_stdev = statistics.stdev(concurrent_ttfb_times) if len(concurrent_ttfb_times) > 1 else 0
                print(f"Concurrent TTFB - Mean: {ttfb_mean:.3f}s, StdDev: {ttfb_stdev:.3f}s, Min: {min(concurrent_ttfb_times):.3f}s, Max: {max(concurrent_ttfb_times):.3f}s")
            
            if concurrent_total_times:
                total_mean = statistics.mean(concurrent_total_times)
                total_stdev = statistics.stdev(concurrent_total_times) if len(concurrent_total_times) > 1 else 0
                print(f"Concurrent Total Time - Mean: {total_mean:.3f}s, StdDev: {total_stdev:.3f}s, Min: {min(concurrent_total_times):.3f}s, Max: {max(concurrent_total_times):.3f}s")
            
            # Compare sequential vs concurrent TTFB
            if ttfb_times and concurrent_ttfb_times:
                seq_ttfb_mean = statistics.mean(ttfb_times)
                conc_ttfb_mean = statistics.mean(concurrent_ttfb_times)
                print(f"\nTTFB Comparison:")
                print(f"  Sequential: {seq_ttfb_mean:.3f}s")
                print(f"  Concurrent: {conc_ttfb_mean:.3f}s")
                print(f"  Difference: {conc_ttfb_mean - seq_ttfb_mean:+.3f}s ({((conc_ttfb_mean/seq_ttfb_mean - 1) * 100):+.1f}%)")

            print(f"\nAudio files saved in '{OUTPUT_DIR}/'")
        
        else:
            print("\nNo successful concurrent runs completed.")

if __name__ == "__main__":
    main() 