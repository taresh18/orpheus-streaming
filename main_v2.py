from src.logger import setup_logger
setup_logger()

import time
from typing import List
from src.vllm_client import VLLMClient
from src.decoder import tokens_decoder
from src.models import TTSRequest, VoiceDetail, VoicesResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
import orjson
from fastapi.responses import JSONResponse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)
    
vllm_client: VLLMClient = None
VOICE_DETAILS: List[VoiceDetail] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and closes resources for the application lifetime."""
    global vllm_client, VOICE_DETAILS
    
    async with VLLMClient() as client:
        vllm_client = client
        
        # Dynamically generate voice details from the loaded client
        VOICE_DETAILS = [
            VoiceDetail(
                name=voice,
                description=f"A standard {voice} voice.",
                language="en",
                gender="unknown",
                accent="american"
            ) for voice in vllm_client.available_voices
        ]
        
        logger.info("Application startup complete. VLLM client is ready.")
        yield 
    
    logger.info("Application shutdown complete. VLLM client has been closed.")


app = FastAPI(
    lifespan=lifespan,
    json_encoder=orjson.dumps,
    json_decoder=orjson.loads
)

@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion using VLLM serve.
    This endpoint is optimized for low latency (Time to First Byte) with concurrent requests.
    """
    start_time = time.perf_counter()

    async def generate_audio_stream():
        first_chunk = True
        try:
            logger.info(f"Generating audio for input: {data.input} with voice: {data.voice}")
            # Generate tokens using VLLM serve
            token_generator = vllm_client.generate_tokens(
                prompt=data.input,
                voice=data.voice,
            )
            
            # Decode tokens to audio
            audio_generator = tokens_decoder(token_generator)

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                yield chunk
        except asyncio.TimeoutError:
            logger.error("Request timed out during audio generation")
            raise
        except Exception as e:
            logger.exception(f"An error occurred during audio generation: {e}")
            raise

    return StreamingResponse(generate_audio_stream(), media_type='audio/pcm')

WS_MSG_DONE = orjson.dumps({"done": True})
WS_MSG_EMPTY_INPUT = orjson.dumps({"type": "no_op", "reason": "empty_input"})

@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.debug("WebSocket connection opened")
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = orjson.loads(raw_data)

            if not data.get("continue", True):
                logger.debug("End of stream message received, closing connection.")
                break

            input_text = data.get("input", "").strip()
            if not input_text:
                logger.debug("Empty or whitespace-only input received, sending no_op.")
                await websocket.send_text(WS_MSG_EMPTY_INPUT.decode('utf-8'))
                continue

            voice = data.get("voice", "tara")
            segment_id = data.get("segment_id", "no_segment_id")
            
            start_msg = orjson.dumps({"type": "start", "segment_id": segment_id})
            end_msg = orjson.dumps({"type": "end", "segment_id": segment_id})
            start_time = time.perf_counter()

            try:
                await websocket.send_text(start_msg.decode('utf-8'))

                logger.info(f"Generating audio for input: '{input_text}' with voice: {voice}")
                
                # Generate tokens using vllm_client
                token_generator = vllm_client.generate_tokens(
                    prompt=input_text,
                    voice=voice,
                )
                
                # Decode tokens to audio
                audio_generator = tokens_decoder(token_generator)

                first_chunk = True
                async for chunk in audio_generator:
                    if first_chunk:
                        ttfb = time.perf_counter() - start_time
                        logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                        first_chunk = False
                    await websocket.send_bytes(chunk)
                
                await websocket.send_text(end_msg.decode('utf-8'))

                if not data.get("continue", True):
                    await websocket.send_text(orjson.dumps({"done": True}).decode('utf-8'))
                    break

            except asyncio.TimeoutError:
                error_msg = orjson.dumps({"error": "Request timed out", "done": True, "segment_id": segment_id})
                logger.error(f"Request timed out for segment '{segment_id}'")
                await websocket.send_text(error_msg.decode('utf-8'))
                break
            except Exception as e:
                error_msg = orjson.dumps({"error": str(e), "done": True, "segment_id": segment_id})
                logger.exception(f"An error occurred for segment '{segment_id}': {e}")
                await websocket.send_text(error_msg.decode('utf-8'))
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.get("/api/voices", response_model=VoicesResponse)
async def get_voices():
    """Get available voices with detailed information."""
    default_voice = vllm_client.available_voices[0] if vllm_client and vllm_client.available_voices else "tara"
    return {
        "voices": VOICE_DETAILS,
        "default": default_voice,
        "count": len(VOICE_DETAILS)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "tts-trt-serve"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090) 