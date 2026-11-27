import json
import pytest
from signal_ai import review


@pytest.mark.parametrize("decision,confidence,available", [
    ("APPROVE", 0.8, True), ("VETO", 0.6, True),
    ("BUY", 0.8, False), ("APPROVE", True, False),
])
def test_model_contract(monkeypatch, decision, confidence, available):
    monkeypatch.setenv("AI_API_KEY", "test")
    monkeypatch.setenv("AI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AI_MODEL", "openai/gpt-oss-20b")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            content = json.dumps(dict(decision=decision, confidence=confidence, commentary="Sample"))
            return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    def post(url, **kwargs):
        assert url == "https://example.invalid/v1/chat/completions"
        body = kwargs["json"]
        assert body["reasoning_effort"] == "low"
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON" in body["messages"][-1]["content"]
        return Response()

    monkeypatch.setattr("signal_ai.requests.post", post)
    result = review({"candidate": "XAUUSD_ORB"})
    assert result["available"] is available
    assert result["decision"] == (decision if available else "VETO")
