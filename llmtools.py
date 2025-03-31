import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator, Iterator


class LLMProvider(ABC):
    """
    Abstract base class for different LLM providers
    """
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]], stream: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Send a completion request to the LLM provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the completion response
        """
        pass
        
    @abstractmethod
    def function_calling(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                        stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Execute a function call with the LLM provider
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the function call response
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.model = model
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def chat_completion(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Implementation of chat completion using OpenAI API
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the completion response
        """
        if not stream:
            # Non-streaming approach
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.8)
            )
            
            # Convert to standard format
            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
                "provider": "openai",
                "model": kwargs.get("model", self.model),
                "raw_response": response
            }
        else:
            # Streaming approach
            stream_response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.8),
                stream=True
            )
            
            # Collect streamed content
            collected_content = ""
            role = None
            finish_reason = None
            
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content += chunk.choices[0].delta.content
                
                # Capture role when it's first available
                if chunk.choices and chunk.choices[0].delta.role and not role:
                    role = chunk.choices[0].delta.role
                
                # Capture finish reason when it's available (in the last chunk)
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # Return with collected content
            return {
                "content": collected_content,
                "role": role or "assistant",  # Default to assistant if not provided
                "finish_reason": finish_reason or "stop",  # Default to stop if not provided
                "provider": "openai",
                "model": kwargs.get("model", self.model),
                "raw_response": None  # Raw response not available with streaming
            }
    
    def function_calling(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Implementation of function calling using OpenAI API
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the function call response
        """
        if not stream:
            # Non-streaming approach
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                tools=tools,
                temperature=kwargs.get("temperature", 0.8),
                tool_choice=kwargs.get("tool_choice", "auto")
            )
            
            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
                "provider": "openai",
                "model": kwargs.get("model", self.model),
                "raw_response": response
            }
            
            # Add tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                result["tool_calls"] = response.choices[0].message.tool_calls
                
            return result
        else:
            # Streaming approach for function calling
            stream_response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                tools=tools,
                temperature=kwargs.get("temperature", 0.8),
                tool_choice=kwargs.get("tool_choice", "auto"),
                stream=True
            )
            
            # Initialize variables to collect streamed data
            collected_content = ""
            role = None
            finish_reason = None
            tool_calls = []
            
            # For accumulating function arguments
            final_tool_calls = {}
            
            for chunk in stream_response:
                # Process content delta if available
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content += chunk.choices[0].delta.content
                
                # Get role when available
                if chunk.choices and chunk.choices[0].delta.role and not role:
                    role = chunk.choices[0].delta.role
                
                # Get finish reason when available (final chunk)
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                
                # Process tool calls if present
                if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        index = tool_call.index
                        
                        # Initialize if first time seeing this tool call
                        if index not in final_tool_calls:
                            final_tool_calls[index] = {
                                "id": tool_call.id or "",
                                "type": tool_call.type or "function",
                                "function": {
                                    "name": tool_call.function.name or "",
                                    "arguments": tool_call.function.arguments or ""
                                }
                            }
                        else:
                            # Append to function name if provided
                            if tool_call.function.name:
                                final_tool_calls[index]["function"]["name"] = tool_call.function.name
                            
                            # Append to arguments if provided
                            if tool_call.function.arguments:
                                final_tool_calls[index]["function"]["arguments"] += tool_call.function.arguments
                            
                            # Set ID if provided and not set
                            if tool_call.id and not final_tool_calls[index]["id"]:
                                final_tool_calls[index]["id"] = tool_call.id
                            
                            # Set type if provided and not set
                            if tool_call.type and not final_tool_calls[index]["type"]:
                                final_tool_calls[index]["type"] = tool_call.type
            
            # Convert accumulated tool calls to list
            final_tool_calls_list = list(final_tool_calls.values())
            
            # Construct result
            result = {
                "content": collected_content,
                "role": role or "assistant",
                "finish_reason": finish_reason or "stop",
                "provider": "openai",
                "model": kwargs.get("model", self.model),
                "raw_response": None
            }
            
            # Add tool calls if any were captured
            if final_tool_calls_list:
                result["tool_calls"] = final_tool_calls_list
                
            return result


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider implementation
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"):
        self.model = model
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    
    def _prepare_anthropic_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Helper method to extract system message and prepare messages for Anthropic API"""
        # Extract system message if present
        system = None
        filtered_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                filtered_messages.append(message)
        
        # Convert to Anthropic message format
        anthropic_messages = []
        for message in filtered_messages:
            # Handle complex content structures (like images in OpenAI)
            if isinstance(message["content"], list):
                # This is a simplified approach - in a real implementation,
                # you would convert OpenAI message format to Anthropic format
                text_parts = []
                for part in message["content"]:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                
                anthropic_messages.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": " ".join(text_parts)}]
                })
            else:
                # Simple text content
                anthropic_messages.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}]
                })
        
        return system, anthropic_messages
    
    def chat_completion(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Implementation of chat completion using Anthropic API
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the completion response
        """
        system, anthropic_messages = self._prepare_anthropic_messages(messages)
        
        if not stream:
            # Non-streaming approach
            response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=anthropic_messages,
                system=system,
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 10000),
            )
            
            # Convert to standard format
            return {
                "content": response.content[0].text if response.content else "",
                "role": "assistant",
                "finish_reason": "stop",  # Anthropic doesn't provide this in the same way
                "provider": "anthropic",
                "model": kwargs.get("model", self.model),
                "raw_response": response
            }
        else:
            # Streaming approach
            stream_response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=anthropic_messages,
                system=system,
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 50000),
                stream=True
            )
            
            # Collect streamed content
            collected_content = ""
            
            for chunk in stream_response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    # Extract text from delta
                    collected_content += chunk.delta.text
            
            # Return with collected content
            return {
                "content": collected_content,
                "role": "assistant",
                "finish_reason": "stop",
                "provider": "anthropic",
                "model": kwargs.get("model", self.model),
                "raw_response": None
            }
    
    def function_calling(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Implementation of function calling for Anthropic
        
        Note: This is a simplified implementation that uses Anthropic's tools API
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the function call response
        """
        system, anthropic_messages = self._prepare_anthropic_messages(messages)
        
        # Convert OpenAI tools to Anthropic tools
        # Note: This is a simplified conversion and may need enhancement for all use cases
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func_def = tool.get("function", {})
                anthropic_tools.append({
                    "name": func_def.get("name", ""),
                    "description": func_def.get("description", ""),
                    "input_schema": func_def.get("parameters", {})
                })
        
        if not stream:
            # Non-streaming approach
            # Call Anthropic API with tools
            response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=anthropic_messages,
                system=system,
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 10000),
                tools=anthropic_tools
            )
            
            # Handle tool selection in the response
            result = {
                "content": response.content[0].text if response.content else "",
                "role": "assistant",
                "finish_reason": "stop",
                "provider": "anthropic",
                "model": kwargs.get("model", self.model),
                "raw_response": response
            }
            
            # Add tool calls if present
            if hasattr(response, 'tool_calls'):
                result["tool_calls"] = response.tool_calls
                
            return result
        else:
            # Streaming approach
            stream_response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=anthropic_messages,
                system=system,
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 50000),
                tools=anthropic_tools,
                stream=True
            )
            
            # Collect streamed content
            collected_content = ""
            tool_calls = []
            
            for chunk in stream_response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    collected_content += chunk.delta.text
                
                # For tools/function calls - implementation will depend on how
                # Anthropic structures tool call responses in their streaming API
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_calls'):
                    # Append tool calls (specifics would depend on Anthropic's API)
                    # This is a placeholder for when Anthropic's function calling API 
                    # with streaming becomes more established
                    if chunk.delta.tool_calls:
                        tool_calls.extend(chunk.delta.tool_calls)
            
            # Construct the result
            result = {
                "content": collected_content,
                "role": "assistant",
                "finish_reason": "stop",
                "provider": "anthropic",
                "model": kwargs.get("model", self.model),
                "raw_response": None
            }
            
            # Add tool calls if any were collected
            if tool_calls:
                result["tool_calls"] = tool_calls
                
            return result


class MultiLLMClient:
    """
    Main client class that abstracts away the provider differences
    """
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the MultiLLM client
        
        Args:
            provider: The provider to use ("openai", "anthropic", etc.)
            api_key: API key for the provider (defaults to environment variable)
            model: Default model to use
        """
        self.provider_name = provider.lower()
        self.api_key = api_key
        self.model = model
        self._setup_provider()
        
    def _setup_provider(self):
        """Set up the selected provider"""
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(
                api_key=self.api_key, 
                model=self.model or "gpt-4o"
            )
        elif self.provider_name == "anthropic":
            self.provider = AnthropicProvider(
                api_key=self.api_key, 
                model=self.model or "claude-3-7-sonnet-20250219"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
    
    def chat_completion(self, messages: List[Dict[str, Any]], stream: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request to the configured provider
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response (default is True to prevent timeouts)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Standardized response dictionary
        """
        return self.provider.chat_completion(messages, stream=stream, **kwargs)
    
    def function_calling(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                       stream: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute a function call with the configured provider
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            stream: Whether to stream the response (default is True to prevent timeouts)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Standardized response dictionary with tool calls if applicable
        """
        return self.provider.function_calling(messages, tools, stream=stream, **kwargs)
    
    def set_provider(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Change the provider on the fly
        
        Args:
            provider: New provider to use
            api_key: Optional new API key
            model: Optional new default model
        """
        self.provider_name = provider.lower()
        if api_key:
            self.api_key = api_key
        if model:
            self.model = model
        self._setup_provider()


# Example usage
if __name__ == "__main__":
    # Initialize with OpenAI
    client = MultiLLMClient(provider="anthropic")
    
    # Simple chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    
    # Using streaming by default
    response = client.chat_completion(messages)
    print(f"OpenAI response: {response['content']}")
    
    # Switch to Anthropic
    client.set_provider("openai")
    
    # Same chat with Anthropic
    response = client.chat_completion(messages)
    print(f"Anthropic response: {response['content']}")
    
    # Function calling example with OpenAI
    client.set_provider("openai")
    
    # Define a weather function
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location","unit"],
                "additionalProperties": False
            },
            "strict": True
        }
    }]
    
    function_messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Boston (temperature in F)?"}
    ]
    
    function_response = client.function_calling(function_messages, tools)
    print(f"Function response: {json.dumps(function_response, default=str, indent=2)}")