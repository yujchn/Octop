"""Unit tests for the ACP passthrough config API (``acp.py`` router).

These tests exercise the two new endpoints directly (bypassing FastAPI's
``Depends`` wiring) using lightweight fakes for the server / registry / user.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from octop.api.routers import acp as acp_router
from octop.infra.errors import ErrorCode, OctopError


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _runners(**overrides: Any) -> dict[str, dict[str, Any]]:
    runners: dict[str, dict[str, Any]] = {
        "hermes": {
            "enabled": True,
            "command": "hermes",
            "args": ["acp"],
            "env": {},
            "trusted": True,
            "tool_parse_mode": "update_detail",
            "stdio_buffer_limit_bytes": 52428800,
        },
        "off": {
            "enabled": False,
            "command": "off",
            "args": [],
            "env": {},
            "trusted": False,
            "tool_parse_mode": "update_detail",
            "stdio_buffer_limit_bytes": 52428800,
        },
    }
    runners.update(overrides)
    return runners


class _FakeAcpSettings:
    def __init__(self, runners: dict[str, dict[str, Any]]) -> None:
        self._runners = runners

    def load_runners(self, user_id: int) -> dict[str, dict[str, Any]]:
        return self._runners


class _FakeRegistry:
    def __init__(
        self,
        config: dict[str, Any],
        runners: dict[str, dict[str, Any]],
    ) -> None:
        self._config = config
        self.acp_settings = _FakeAcpSettings(runners)

    def get_row(self, agent_id: str) -> Any:
        return SimpleNamespace(user_id=1)

    def get_config(self, agent_id: str) -> dict[str, Any]:
        return self._config

    async def update_config_json(self, agent_id: str, config_json: str) -> None:
        self._config.clear()
        self._config.update(json.loads(config_json))


class _FakeServer:
    def __init__(self, registry: _FakeRegistry) -> None:
        self.app_runtime = SimpleNamespace(agent_registry=registry)


def _user() -> Any:
    return SimpleNamespace(id=1, is_admin=False)


def _server(registry: _FakeRegistry) -> _FakeServer:
    return _FakeServer(registry)


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_disable_returns_none() -> None:
    registry = _FakeRegistry({}, _runners())
    assert (
        acp_router._normalize_passthrough(
            acp_router.ACPPassthroughBody(enabled=False, runner="hermes"),
            registry=registry,
            user_id=1,
        )
        is None
    )
    assert (
        acp_router._normalize_passthrough(
            acp_router.ACPPassthroughBody(enabled=True, runner="   "),
            registry=registry,
            user_id=1,
        )
        is None
    )


def test_normalize_enable_strips_and_omits_empty() -> None:
    registry = _FakeRegistry({}, _runners())
    full = acp_router._normalize_passthrough(
        acp_router.ACPPassthroughBody(
            enabled=True,
            runner=" hermes ",
            cwd=" /abs ",
            max_runtime=300,
            start_prompt=" hi ",
        ),
        registry=registry,
        user_id=1,
    )
    assert full == {
        "runner": "hermes",
        "cwd": "/abs",
        "max_runtime": 300,
        "start_prompt": "hi",
    }
    slim = acp_router._normalize_passthrough(
        acp_router.ACPPassthroughBody(enabled=True, runner="hermes"),
        registry=registry,
        user_id=1,
    )
    assert slim == {"runner": "hermes"}


def test_normalize_unknown_runner_raises() -> None:
    registry = _FakeRegistry({}, _runners())
    try:
        acp_router._normalize_passthrough(
            acp_router.ACPPassthroughBody(enabled=True, runner="nope"),
            registry=registry,
            user_id=1,
        )
    except OctopError as exc:
        assert exc.code is ErrorCode.SLASH_BAD_ARGS
    else:
        raise AssertionError("expected OctopError")


def test_normalize_disabled_runner_raises() -> None:
    registry = _FakeRegistry({}, _runners())
    try:
        acp_router._normalize_passthrough(
            acp_router.ACPPassthroughBody(enabled=True, runner="off"),
            registry=registry,
            user_id=1,
        )
    except OctopError as exc:
        assert exc.code is ErrorCode.SLASH_BAD_ARGS
    else:
        raise AssertionError("expected OctopError")


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


async def test_get_passthrough_unconfigured() -> None:
    registry = _FakeRegistry({}, _runners())
    result = await acp_router.get_acp_passthrough(
        "agent-1", user=_user(), server=_server(registry)
    )
    assert result == {"acp_passthrough": None}


async def test_put_then_get_roundtrip_preserves_other_keys() -> None:
    cfg: dict[str, Any] = {"acp": {"tool_enabled": True}, "other": {"keep": 1}}
    registry = _FakeRegistry(cfg, _runners())
    server = _server(registry)
    result = await acp_router.put_acp_passthrough(
        "agent-1",
        acp_router.ACPPassthroughBody(
            enabled=True, runner="hermes", cwd="/abs", max_runtime=300, start_prompt="hi"
        ),
        user=_user(),
        server=server,
    )
    assert result == {
        "acp_passthrough": {
            "runner": "hermes",
            "cwd": "/abs",
            "max_runtime": 300,
            "start_prompt": "hi",
        }
    }
    # Other config keys are preserved untouched.
    assert cfg["acp"] == {"tool_enabled": True}
    assert cfg["other"] == {"keep": 1}

    got = await acp_router.get_acp_passthrough("agent-1", user=_user(), server=server)
    assert got["acp_passthrough"]["runner"] == "hermes"


async def test_put_disable_removes_key() -> None:
    cfg: dict[str, Any] = {
        "acp": {"tool_enabled": True},
        "acp_passthrough": {"runner": "hermes"},
    }
    registry = _FakeRegistry(cfg, _runners())
    server = _server(registry)
    result = await acp_router.put_acp_passthrough(
        "agent-1",
        acp_router.ACPPassthroughBody(enabled=False, runner="hermes"),
        user=_user(),
        server=server,
    )
    assert result == {"acp_passthrough": None}
    assert "acp_passthrough" not in cfg
    assert cfg["acp"] == {"tool_enabled": True}


async def test_put_invalid_runner_raises_and_no_write() -> None:
    cfg: dict[str, Any] = {}
    registry = _FakeRegistry(cfg, _runners())
    server = _server(registry)
    try:
        await acp_router.put_acp_passthrough(
            "agent-1",
            acp_router.ACPPassthroughBody(enabled=True, runner="nope"),
            user=_user(),
            server=server,
        )
    except OctopError as exc:
        assert exc.code is ErrorCode.SLASH_BAD_ARGS
    else:
        raise AssertionError("expected OctopError")
    assert cfg == {}
