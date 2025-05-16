import typing
from contextlib import contextmanager
from functools import wraps
from openai.types.responses.response_usage import ResponseUsage

@contextmanager
def OpenAIUsage(client) -> typing.Generator[typing.Dict[str, float], None, None]:
    """
    Context manager for measuring OpenAI API usage.
    Overwrites client's responses.create and responses.parse to collect usage metrics.
    """
    usage: typing.Dict[str, float] = {}
    # Store original methods
    original_create = client.responses.create
    original_parse = client.responses.parse

    def _flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            usage_data = getattr(response, 'usage', None)
            flat_usage = {}
            if isinstance(usage_data, ResponseUsage):
                flat_usage = _flatten_dict(usage_data.model_dump())
            elif isinstance(usage_data, dict):
                flat_usage = _flatten_dict(usage_data)
            if flat_usage:
                for k, v in flat_usage.items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
            return response
        return wrapper

    # Override methods
    client.responses.create = _wrap(original_create)
    client.responses.parse = _wrap(original_parse)

    try:
        yield usage
    finally:
        # Restore original methods
        client.responses.create = original_create
        client.responses.parse = original_parse
        print("OpenAI Usage:", usage)