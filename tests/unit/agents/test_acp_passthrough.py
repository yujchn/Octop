"""Unit tests for agent-level ACP passthrough (experimental)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import pytest

from octop.infra.agents import acp_passthrough as acp_mod
from octop.infra.agents import manager as manager_module
from octop.infra.agents.acp_passthrough import (
    extract_turn_text,
    resolve_passthrough_config,
    stream_acp_passthrough,
)
from octop.infra.agents.manager import AgentManager

# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------


def _hermes_runner(*, enabled: bool = True) -> dict[str, dict[str, Any]]:
    return {
        "hermes": {
            "enabled": enabled,
            "command": "hermes",
            "args": ["acp"],
            "env": {},
            "trusted": True,
            "tool_parse_mode": "update_detail",
            "stdio_buffer_limit_bytes": 52428800,
        }
    }


class _FakeRow:
    def __init__(self, user_id: int = 1) -> None:
        self.user_id = user_id


class _FakeAcpSettings:
    def __init__(self, runners: dict[str, dict[str, Any]]) -> None:
        self._runners = runners

    def load_runners(self, user_id: int) -> dict[str, dict[str, Any]]:
        return self._runners


class _FakePassthroughManager:
    def __init__(self, runners: dict[str, dict[str, Any]], workspace: str = "/tmp/ws") -> None:
        self._runners = runners
        self._workspace = Path(workspace)
        self._connector_user_override: dict[str, int] = {}

    def get_row(self, agent_id: str) -> _FakeRow:
        return _FakeRow(1)

    @property
    def acp_settings(self) -> _FakeAcpSettings:
        return _FakeAcpSettings(self._runners)

    def resolve_workspace_dir(self, agent_id: str) -> Path:
        return self._workspace


class _FakeSession:
    def __init__(self, acp_session_id: str = "acp-sess-1") -> None:
        self.acp_session_id = acp_session_id


class _FakeService:
    def __init__(self, *, pending: bool = False, status: str = "completed") -> None:
        self.pending = pending
        self.status = status
        self.run_turn_calls: list[tuple[Any, ...]] = []
        self.resume_calls: list[tuple[Any, ...]] = []
        self.canceled = False

    async def get_pending_permission(self, *, thread_id: str, runner: str) -> Any | None:
        return object() if self.pending else None

    async def get_session(self, thread_id: str, runner: str) -> _FakeSession:
        return _FakeSession()

    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        self.run_turn_calls.append((kwargs["thread_id"], kwargs["runner"], kwargs["prompt_blocks"]))
        # Tool/status events stream through on_message; assistant text comes back
        # via the final event (faithful to ACPHostedClient).
        await kwargs["on_message"]({"type": "tool_start", "kind": "bash", "detail": "ls"}, True)
        await kwargs["on_message"]({"type": "status", "status": "run_finished"}, True)
        if self.status == "permission_required":
            return {"status": "permission_required", "suspended_permission": None}
        return {"status": "completed", "event": {"type": "text", "text": "hello"}}

    async def resume_permission(self, **kwargs: Any) -> dict[str, Any]:
        self.resume_calls.append((kwargs["acp_session_id"], kwargs["option_id"]))
        return {"status": "completed", "event": {"type": "text", "text": "resumed"}}

    async def cancel_turn(self, *, thread_id: str, runner: str) -> bool:
        self.canceled = True
        return True


async def _collect(agen: Any) -> list[dict[str, Any]]:
    return [chunk async for chunk in agen]


# ---------------------------------------------------------------------------
# config gate + turn text extraction
# ---------------------------------------------------------------------------


def test_resolve_passthrough_config_unconfigured() -> None:
    assert resolve_passthrough_config({}) is None
    assert resolve_passthrough_config({"acp_passthrough": {}}) is None
    assert resolve_passthrough_config({"acp_passthrough": {"runner": ""}}) is None


def test_resolve_passthrough_config_normalizes() -> None:
    cfg = resolve_passthrough_config({"acp_passthrough": {"runner": " hermes "}})
    assert cfg is not None
    assert cfg["runner"] == "hermes"


def test_extract_turn_text_variants() -> None:
    assert extract_turn_text({"content": "plain"}) == "plain"
    assert (
        extract_turn_text({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        == "a\nb"
    )
    assert extract_turn_text({"messages": [{"role": "user", "content": "m1"}]}) == "m1"

    class _HM:
        type = "human"
        content = [{"type": "text", "text": "hm"}]

    assert extract_turn_text({"messages": [_HM()]}) == "hm"
    # last human message wins
    assert (
        extract_turn_text(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "skip"},
                    {"role": "user", "content": "last"},
                ]
            }
        )
        == "last"
    )


# ---------------------------------------------------------------------------
# passthrough stream behaviour (module level)
# ---------------------------------------------------------------------------


async def test_stream_passthrough_yields_tokens_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService()
    monkeypatch.setattr(acp_mod, "get_acp_service", lambda key: None)
    monkeypatch.setattr(acp_mod, "init_acp_service", lambda key, config: service)

    mgr = _FakePassthroughManager(_hermes_runner())
    chunks = await _collect(
        stream_acp_passthrough(
            mgr,
            "agent-1",
            "thread-1",
            {"content": "turn text"},
            {"runner": "hermes"},
        )
    )

    assert service.run_turn_calls
    tokens = [c for c in chunks if c.get("type") == "token"]
    assert tokens and any("hello" in t["content"] for t in tokens)
    assert chunks[-1] == {"type": "done"}
    assert all(c.get("type") != "error" for c in chunks)


async def test_stream_passthrough_runner_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService()
    monkeypatch.setattr(acp_mod, "get_acp_service", lambda key: None)
    monkeypatch.setattr(acp_mod, "init_acp_service", lambda key, config: service)

    mgr = _FakePassthroughManager(_hermes_runner(enabled=False))
    chunks = await _collect(
        stream_acp_passthrough(
            mgr,
            "agent-1",
            "thread-1",
            {"content": "turn text"},
            {"runner": "hermes"},
        )
    )

    assert chunks[0]["type"] == "error"
    assert "hermes" in chunks[0]["message"]
    assert chunks[-1] == {"type": "done"}
    assert service.run_turn_calls == []


async def test_stream_passthrough_resume_pending_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeService(pending=True)
    monkeypatch.setattr(acp_mod, "get_acp_service", lambda key: None)
    monkeypatch.setattr(acp_mod, "init_acp_service", lambda key, config: service)

    mgr = _FakePassthroughManager(_hermes_runner())
    chunks = await _collect(
        stream_acp_passthrough(
            mgr,
            "agent-1",
            "thread-1",
            {"content": "opt-2"},
            {"runner": "hermes"},
        )
    )

    assert service.resume_calls == [("acp-sess-1", "opt-2")]
    assert service.run_turn_calls == []
    tokens = [c for c in chunks if c.get("type") == "token"]
    assert any("resumed" in t["content"] for t in tokens)
    assert chunks[-1] == {"type": "done"}


# ---------------------------------------------------------------------------
# AgentManager.stream routing
# ---------------------------------------------------------------------------


class _FakeHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def stream(self, agent_id: str, req: dict[str, Any]) -> Any:
        self.calls.append((agent_id, req))
        yield {"type": "token", "content": "harness-reply"}
        yield {"type": "done"}

    def cancel(self, agent_id: str, thread_id: str) -> None:
        pass


@asynccontextmanager
async def _noop_track(agent_id: str) -> Any:
    yield


def _make_manager(config: dict[str, Any], harness: _FakeHarness) -> AgentManager:
    mgr = object.__new__(AgentManager)
    lock = asyncio.Lock()
    mgr._harness_manager = harness
    mgr._connector_user_override = {}
    mgr.get_config = lambda agent_id: config  # type: ignore[method-assign]
    mgr._prepare_stream_request = lambda agent_id, request: dict(request)  # type: ignore[method-assign]
    mgr._apply_pending_bootstrap_graph_refresh = lambda agent_id: None  # type: ignore[method-assign]
    mgr._thread_execution_lock = lambda agent_id, thread_id: lock  # type: ignore[method-assign]
    mgr._track_invocation = _noop_track  # type: ignore[method-assign]
    mgr._unavailable_error = lambda agent_id: RuntimeError("unavailable")  # type: ignore[method-assign]
    return mgr


async def test_stream_routing_passthrough_skips_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_passthrough(manager: Any, agent_id: str, thread_id: str, request: Any, cfg: Any) -> Any:
        yield {"type": "token", "content": "acp-reply"}
        yield {"type": "done"}

    monkeypatch.setattr(manager_module, "stream_acp_passthrough", fake_passthrough)

    harness = _FakeHarness()
    mgr = _make_manager({"acp_passthrough": {"runner": "hermes"}}, harness)
    chunks = await _collect(mgr.stream("agent-1", {"thread_id": "thread-1"}))

    assert harness.calls == []
    assert [c for c in chunks if c.get("type") == "token"] == [
        {"type": "token", "content": "acp-reply"}
    ]
    assert chunks[-1] == {"type": "done"}


async def test_stream_routing_unconfigured_uses_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "hitl_thread_scope", nullcontext)
    monkeypatch.setattr(
        manager_module, "thread_id_from_request", lambda req: req.get("thread_id") or ""
    )

    harness = _FakeHarness()
    mgr = _make_manager({}, harness)
    chunks = await _collect(mgr.stream("agent-1", {"thread_id": "thread-1"}))

    assert len(harness.calls) == 1
    tokens = [c for c in chunks if c.get("type") == "token"]
    assert tokens == [{"type": "token", "content": "harness-reply"}]
    assert chunks[-1] == {"type": "done"}
