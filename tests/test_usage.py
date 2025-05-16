from openai import OpenAI

from openai_usage.usage import OpenAIUsage

class DummyResponse:
    def __init__(self, usage):
        self.usage = usage

class DummyResponses:
    def create(self, *args, **kwargs):
        return DummyResponse({"token1": 1.0})
    def parse(self, *args, **kwargs):
        return DummyResponse({"token2": 2.0})

class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()

def test_openai_usage_collects_create_and_parse_usage():
    client = DummyClient()
    original_create = client.responses.create
    original_parse = client.responses.parse

    with OpenAIUsage(client) as usage:
        resp_create = client.responses.create(prompt="hello")
        resp_parse = client.responses.parse(data="world")
        assert resp_create.usage == {"token1": 1.0}
        assert resp_parse.usage == {"token2": 2.0}
        assert usage == {"token1": 1.0, "token2": 2.0}

    # After context exit, methods should be restored to original
    assert client.responses.create == original_create
    assert client.responses.parse == original_parse
def test_openai_usage_nested_usage():
    class NestedDummyResponse:
        def __init__(self, usage):
            self.usage = usage

    class NestedDummyResponses:
        def create(self, *args, **kwargs):
            # Simulate nested usage structure
            return NestedDummyResponse({
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 27,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 38
            })
        def parse(self, *args, **kwargs):
            return NestedDummyResponse({
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 15
            })

    class NestedDummyClient:
        def __init__(self):
            self.responses = NestedDummyResponses()

    client = NestedDummyClient()
    with OpenAIUsage(client) as usage:
        client.responses.create()
        client.responses.parse()
        # Check flattened keys
        assert usage["input_tokens"] == 16
        assert usage["input_tokens_details.cached_tokens"] == 2
        assert usage["output_tokens"] == 37
        assert usage["output_tokens_details.reasoning_tokens"] == 1
        assert usage["total_tokens"] == 53
    
def test_openai_client_usage_collection():
    openai_client = OpenAI()
    
    with OpenAIUsage(openai_client) as usage:
        resp_create = openai_client.responses.create(input="Tell me a joke", model="gpt-3.5-turbo")
        resp_create2 = openai_client.responses.create(input="Tell me second joke", model="gpt-3.5-turbo")
        # The usage dict should match the flattened keys from the response usage
        expected_keys = [
            "input_tokens",
            "input_tokens_details.cached_tokens",
            "output_tokens",
            "output_tokens_details.reasoning_tokens",
            "total_tokens"
        ]
        for key in expected_keys:
            assert key in usage
        resp_usage = getattr(resp_create, "usage", None)
        resp_usage2 = getattr(resp_create2, "usage", None)
        assert resp_usage is not None
        assert resp_usage2 is not None
        assert usage["input_tokens"] == resp_usage.input_tokens + resp_usage2.input_tokens
        assert usage["input_tokens_details.cached_tokens"] == resp_usage.input_tokens_details.cached_tokens + resp_usage2.input_tokens_details.cached_tokens
        assert usage["output_tokens"] == resp_usage.output_tokens + resp_usage2.output_tokens
        assert usage["output_tokens_details.reasoning_tokens"] == resp_usage.output_tokens_details.reasoning_tokens + resp_usage2.output_tokens_details.reasoning_tokens
        assert usage["total_tokens"] == resp_usage.total_tokens + resp_usage2.total_tokens