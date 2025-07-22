import aiohttp
import asyncio
import orjson
import logging
import os
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for making requests to VLLM serve server."""
    
    def __init__(self):
        """Initializes the VLLM client"""
        try:
            self.server_url = f"{os.environ['SERVE_URL']}/completions"
            self.model_name = os.environ["MODEL_NAME"]
            
            max_conns_per_worker = int(os.environ["MAX_CONNS_PER_WORKER"])
            self.timeout = aiohttp.ClientTimeout(
                total=float(os.environ["TOTAL_TIMEOUT"]),
                connect=float(os.environ["CONNECT_TIMEOUT"]),
                sock_read=float(os.environ["READ_TIMEOUT"])
            )

            api_key = os.environ["API_KEY"]
            self.available_voices = [v.strip() for v in os.environ["AVAILABLE_VOICES"].split(',')]
            
            self.sampling_params = {
                "max_tokens": int(os.environ["MAX_TOKENS"]),
                "temperature": float(os.environ["TEMPERATURE"]),
                "top_p": float(os.environ["TOP_P"]),
                "repetition_penalty": float(os.environ["REPETITION_PENALTY"]),
                "stop_token_ids": [int(t.strip()) for t in os.environ["STOP_TOKEN_IDS"].split(',')]
            }
            
            self.max_retries = int(os.environ["MAX_RETRIES"])
            self.retry_delay = float(os.environ["RETRY_DELAY"])

        except KeyError as e:
            logger.error(f"Missing a critical environment variable: {e}")
            raise ValueError(f"Missing a critical environment variable: {e}") from e

        # Session and Connector
        self._connector = aiohttp.TCPConnector(
            limit_per_host=max_conns_per_worker,
            force_close=False,               
            enable_cleanup_closed=False,        
            ssl=False                           
        )
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream"
        }
        self._session: aiohttp.ClientSession = None
        logger.info(f"VLLMClient initialized for server: {self.server_url}")

    async def __aenter__(self):
        """Creates the aiohttp session when entering the async context."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=self._connector,
                headers=self._headers,
                json_serialize=lambda obj: orjson.dumps(obj).decode('utf-8')
            )
        logger.info("VLLMClient session started.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the aiohttp session when exiting the async context."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("VLLMClient session closed.")
    
    def _format_prompt(self, prompt: str, voice: str) -> str:
        """Formats the prompt for the VLLM API."""
        adapted_prompt = f"{voice}: {prompt}"
        return f"<custom_token_3><|begin_of_text|>{adapted_prompt}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    
    async def generate_tokens(self, prompt: str, voice: str) -> AsyncGenerator[str, None]:
        """
        Generates tokens asynchronously from the VLLM server with connection pooling and retries.
        """
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' is not available. Available voices: {self.available_voices}")
        
        formatted_prompt = self._format_prompt(prompt, voice)
        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "stream": True,
            **self.sampling_params
        }
                
        for attempt in range(self.max_retries + 1):
            try:
                # Use the persistent session to make the request
                async with self._session.post(self.server_url, json=payload, timeout=self.timeout) as response:
                    response.raise_for_status()
                    
                    # Process the streaming response
                    buffer = ""
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            buffer += line_str + '\n'
                            
                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if line.startswith('data: '):
                                    data_str = line[6:]
                                    
                                    if data_str.strip() == '[DONE]':
                                        return # Successful completion
                                    
                                    try:
                                        data = orjson.loads(data_str)
                                        if (choices := data.get('choices')) and choices[0].get('text'):
                                            yield choices[0]['text']
                                    except (orjson.JSONDecodeError, IndexError, KeyError):
                                        logger.warning(f"Could not parse chunk: {data_str}")
                                        continue
                return # Exit loop on success
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Request to VLLM failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff
                else:
                    logger.error("VLLM request failed after all retries.")
                    raise Exception("Failed to get response from VLLM server.") from e