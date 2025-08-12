"""Ollama client for local LLM integration."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Configuration for an Ollama model."""
    name: str
    size: Optional[str] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = -1
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
    
    @asynccontextmanager
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        
        try:
            yield self._session
        finally:
            # Session will be closed when client is closed
            pass
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if service is reachable, False otherwise
        """
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama service not available: {e}")
            return False
    
    async def list_models(self) -> List[OllamaModel]:
        """List available models.
        
        Returns:
            List of available Ollama models
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to list models: {response.status}")
                
                data = await response.json()
                models = []
                
                for model_data in data.get("models", []):
                    model = OllamaModel(
                        name=model_data.get("name", ""),
                        size=model_data.get("size"),
                        digest=model_data.get("digest"),
                        modified_at=model_data.get("modified_at"),
                        details=model_data.get("details", {})
                    )
                    models.append(model)
                
                return models
    
    async def pull_model(
        self, 
        model_name: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Pulling model: {model_name}")
        
        payload = {"name": model_name}
        
        try:
            async with self._get_session() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to pull model {model_name}: {response.status}")
                        return False
                    
                    # Stream progress updates
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode())
                                if progress_callback and "status" in data:
                                    progress_callback(data["status"])
                                
                                # Check if completed
                                if data.get("status") == "success":
                                    logger.info(f"Successfully pulled model: {model_name}")
                                    return True
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_text(
        self,
        model: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using specified model.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt
            config: Generation configuration
            stream: Whether to stream the response
            
        Returns:
            Generated text or async generator for streaming
        """
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.num_predict,
                "repeat_penalty": config.repeat_penalty,
            }
        }
        
        # Add optional parameters
        if config.seed is not None:
            payload["options"]["seed"] = config.seed
        if config.stop:
            payload["options"]["stop"] = config.stop
        
        if stream:
            return self._generate_stream(payload)
        else:
            return await self._generate_single(payload)
    
    async def _generate_single(self, payload: Dict[str, Any]) -> str:
        """Generate single response."""
        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Generation failed: {response.status}")
                
                data = await response.json()
                return data.get("response", "")
    
    async def _generate_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Generation failed: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if "response" in data:
                                yield data["response"]
                            
                            # Check if completed
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat with the model using conversation history.
        
        Args:
            model: Name of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            config: Generation configuration  
            stream: Whether to stream the response
            
        Returns:
            Generated response or async generator for streaming
        """
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.num_predict,
                "repeat_penalty": config.repeat_penalty,
            }
        }
        
        # Add optional parameters
        if config.seed is not None:
            payload["options"]["seed"] = config.seed
        if config.stop:
            payload["options"]["stop"] = config.stop
        
        if stream:
            return self._chat_stream(payload)
        else:
            return await self._chat_single(payload)
    
    async def _chat_single(self, payload: Dict[str, Any]) -> str:
        """Single chat response."""
        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Chat failed: {response.status}")
                
                data = await response.json()
                message = data.get("message", {})
                return message.get("content", "")
    
    async def _chat_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming chat response."""
        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Chat failed: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            message = data.get("message", {})
                            if "content" in message:
                                yield message["content"]
                            
                            # Check if completed
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            models = await self.list_models()
            for model in models:
                if model.name == model_name:
                    return {
                        "name": model.name,
                        "size": model.size,
                        "digest": model.digest,
                        "modified_at": model.modified_at,
                        "details": model.details
                    }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from local storage.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"name": model_name}
            
            async with self._get_session() as session:
                async with session.delete(
                    f"{self.base_url}/api/delete",
                    json=payload
                ) as response:
                    success = response.status == 200
                    if success:
                        logger.info(f"Successfully deleted model: {model_name}")
                    else:
                        logger.error(f"Failed to delete model {model_name}: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
