"""Tests for the real HTTP smoke orchestration without opening a socket."""

from __future__ import annotations

import json

from app.eval import http_smoke


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.requests = []

    def get(self, path):
        return self.request("GET", path)

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/health":
            return FakeResponse(
                body={
                    "status": "healthy",
                    "capabilities": {"hybrid_search": True},
                }
            )
        if path == "/openapi.json":
            return FakeResponse(body={"paths": {"/health": {}, "/search": {}, "/ask": {}}})
        if path == "/":
            return FakeResponse(text="<title>CiteQuest</title>")
        if path == "/search":
            mode = kwargs["json"]["mode"]
            return FakeResponse(
                body={
                    "mode": mode,
                    "effective_alpha": 0.5 if mode == "hybrid" else None,
                    "latency_ms": 12.5,
                    "results": [
                        {
                            "paper_id": "P1",
                            "chunk_id": "P1_chunk0",
                            "title": "Paper one",
                            "year": 2025,
                            "venue": "arXiv",
                            "authors": [],
                            "score": 1.0,
                            "snippet": "evidence",
                            "abstract": "evidence",
                        }
                    ],
                }
            )
        if path == "/ask":
            return FakeResponse(
                body={
                    "answer": "Grounded answer [1].",
                    "effective_alpha": 0.5,
                    "citations": [{"citation_id": 1}],
                    "citation_valid": True,
                    "latency_ms": 3.0,
                }
            )
        raise AssertionError(f"Unexpected request: {method} {path}")

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _artifacts(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    db_path.write_bytes(b"db")
    (index_dir / "index.faiss").write_bytes(b"index")
    (index_dir / "id_map.json").write_text("[]", encoding="utf-8")
    (index_dir / "build_meta.json").write_text("{}", encoding="utf-8")
    return db_path, index_dir


def test_real_http_smoke_orchestration_passes_and_stops_server(tmp_path, monkeypatch):
    db_path, index_dir = _artifacts(tmp_path)
    process = FakeProcess()
    client = FakeClient()
    monkeypatch.setattr(http_smoke, "_port_available", lambda *args: True)
    monkeypatch.setattr(http_smoke, "_git_value", lambda *args: "test-value")
    monkeypatch.setattr(http_smoke.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(
        http_smoke,
        "collect_accelerator_info",
        lambda: {"device_name": "Test GPU"},
    )

    def fake_popen(*args, **kwargs):
        process.args = args
        process.kwargs = kwargs
        return process

    monkeypatch.setattr(http_smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(http_smoke.httpx, "Client", lambda *args, **kwargs: client)

    report = http_smoke.run_http_smoke(
        db_path=db_path,
        index_dir=index_dir,
        log_path=tmp_path / "server.log",
        port=18765,
    )

    assert report["status"] == "passed"
    assert report["endpoints"]["openapi"]["passed"] is True
    assert report["endpoints"]["search_hybrid"]["passed"] is True
    assert report["endpoints"]["ask_mock_rag"]["citation_valid"] is True
    assert report["configuration"]["external_llm_called"] is False
    assert report["server"]["stopped"] is True
    assert process.terminated is True
    assert process.kwargs["env"]["LLM_API_KEY"] == ""
    assert process.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert process.kwargs["env"]["CITEQUEST_HYBRID_ALPHA"] == "0.5"
    hybrid_request = next(
        kwargs["json"]
        for method, path, kwargs in client.requests
        if method == "POST" and path == "/search" and kwargs["json"]["mode"] == "hybrid"
    )
    assert "alpha" not in hybrid_request
    assert client.closed is True


def test_http_smoke_fails_before_start_when_artifacts_are_missing(tmp_path):
    report = http_smoke.run_http_smoke(
        db_path=tmp_path / "missing.sqlite",
        index_dir=tmp_path / "missing-faiss",
        log_path=tmp_path / "server.log",
    )

    assert report["status"] == "failed"
    assert "Missing required artifacts" in report["errors"][0]
    assert report["endpoints"] == {}


def test_http_smoke_outputs_are_explicitly_non_benchmark(tmp_path):
    report = {
        "status": "passed",
        "created_at": "2026-08-20T00:00:00+00:00",
        "configuration": {"base_url": "http://127.0.0.1:8765"},
        "server": {"stopped": True},
        "endpoints": {"health": {"passed": True, "status_code": 200}},
        "errors": [],
    }
    json_path = tmp_path / "http.json"
    markdown_path = tmp_path / "http.md"

    http_smoke.write_http_smoke_outputs(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "passed"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "real subprocess" in markdown
    assert "not Retrieval Benchmark v1" in markdown


def test_static_frontend_mount_is_registered_after_health_route():
    from app.main import app

    paths = [getattr(route, "path", None) for route in app.routes]

    assert paths.index("/health") < paths.index("")
