import typing
from contextlib import contextmanager
from functools import wraps
import inspect
from openai.types.responses.response_usage import ResponseUsage
from .model_costs import MODEL_COST_PER_1K_TOKENS

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
            # Extract model argument from args/kwargs
            model_name = None
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if "model" in kwargs:
                    model_name = kwargs["model"]
                elif "model" in params:
                    idx = params.index("model")
                    if idx < len(args):
                        model_name = args[idx]
            except Exception:
                pass  # fallback: model_name remains None if extraction fails

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

                # Cost calculation (separate for input, input cached, output, and total)
                if model_name:
                    # Always initialize all cost keys to 0.0 for consistency
                    cost_input = 0.0
                    cost_input_cached = 0.0
                    cost_output = 0.0

                    for usage_key, token_count in flat_usage.items():
                        if not isinstance(token_count, (int, float)):
                            continue
                        # Input tokens
                        if usage_key in ("prompt_tokens", "input_tokens"):
                            cost_key = model_name
                            per_1k = MODEL_COST_PER_1K_TOKENS.get(cost_key)
                            if per_1k:
                                cost_input += (token_count / 1000.0) * per_1k
                        # Input cached tokens
                        elif usage_key in ("input_cached_tokens",):
                            cost_key = f"{model_name}-cached"
                            per_1k = MODEL_COST_PER_1K_TOKENS.get(cost_key)
                            if per_1k:
                                cost_input_cached += (token_count / 1000.0) * per_1k
                        # Output tokens
                        elif usage_key in ("completion_tokens", "output_tokens"):
                            cost_key = f"{model_name}-completion"
                            per_1k = MODEL_COST_PER_1K_TOKENS.get(cost_key)
                            if per_1k:
                                cost_output += (token_count / 1000.0) * per_1k
                        # else: skip unknown usage keys

                    cost_total = cost_input + cost_input_cached + cost_output

                    # Always set all cost keys, even if zero
                    usage["cost_input_tokens"] = usage.get("cost_input_tokens", 0) + cost_input
                    usage["cost_input_cached_tokens"] = usage.get("cost_input_cached_tokens", 0) + cost_input_cached
                    usage["cost_output_tokens"] = usage.get("cost_output_tokens", 0) + cost_output
                    usage["cost_total"] = usage.get("cost_total", 0) + cost_total
                    # For backward compatibility, also set "cost" as total
                    usage["cost"] = usage.get("cost", 0) + cost_total
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