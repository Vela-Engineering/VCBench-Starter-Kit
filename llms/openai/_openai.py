from openai import OpenAI
from core import settings


class OpenAIProvider():
    """Minimal OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (uses settings.OPENAI_ORG_API_KEY if not provided)
            model: Model to use for generation
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def get_llm_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Get a response from the OpenAI API.
        
        Args:
            system_prompt: The system prompt that sets the context
            user_prompt: The user's specific question or request
            temperature: Controls randomness in the response (0.0 to 2.0)
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            The model's response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content


# Backward compatibility function
def get_llm_response(system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
    """Legacy function for backward compatibility."""
    provider = OpenAIProvider()
    return provider.get_llm_response(system_prompt, user_prompt, temperature)