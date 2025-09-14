from .openai import OpenAIProvider


def get_llm_provider(provider: str, **kwargs):
    """
    Get an LLM provider instance.
    
    Args:
        provider: The provider name (e.g., 'openai')
        **kwargs: Additional arguments to pass to the provider
        
    Returns:
        An instance of the requested provider
    """
    if provider.lower() == "openai":
        return OpenAIProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = ['get_llm_provider', 'OpenAIProvider']
