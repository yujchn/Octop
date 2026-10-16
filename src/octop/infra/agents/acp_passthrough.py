"""Agent-level ACP passthrough (experimental).

When an agent's ``config_json`` carries an ``acp_passthrough`` block, its turns
are forwarded straight to an external ACP runner (hermes, openclaw-*, …) instead
of the harness LLM. ACP events are converted back into the Octop chunk protocol
(``token`` / ``error`` / ``done``) so every channel keeps working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from octop_harness.acp.models import ACPConfig
from octop_harness.acp.service import get_acp_service, init_acp_service
from octop_harness.acp.tool_adapter import format_permission_required, render_event_text

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RUNTIME = 300.0
_DEFAULT_START_PROMPT = "hi"

# Queue sentinel emitted once the ACP turn settles, so the consumer can drain
# without polling.
_SENTINEL = object()


def resolve_passthrough_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return the normalized ``acp_passthrough`` block, or None when not opted in.

    A block without a non-empty ``runner`` is treated as unconfigured so the
    original harness path stays untouched for everyone else.
    """
    raw = cfg.get("acp_passthrough")
    if not isinstance(raw, dict):
        return None
    runner = str(raw.get("runner") or "").strip()
    if not runner:
        return None
    normalized = dict(raw)
    normalized["runner"] = runner
    return normalized


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part.strip())
    return ""


def extract_turn_text(request: dict[str, Any]) -> str:
    """Extract this turn's user text from a ``build_harness_request`` request.

    Covers ``content`` as ``str`` or ``list[dict]`` text blocks, and
    ``messages`` entries that are either dicts or langchain ``HumanMessage``
    objects (taking the most recent human turn).
    """
    content = request.get("content")
    if content is not None:
        text = _content_text(content)
        if text.strip():
            return text

    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = str(msg.get("role") or "").lower()
                if role in ("user", "human"):
                    text = _content_text(msg.get("content"))
                    if text.strip():
                        return text
            else:
                mtype = str(getattr(msg, "type", "") or "").lower()
                if mtype in ("user", "human"):
                    text = _content_text(getattr(msg, "content", None))
                    if text.strip():
                        return text
    return ""


def _build_acp_config(manager: Any, agent_id: str) -> ACPConfig:
    """Resolve the agent owner's runner pool into an :class:`ACPConfig`."""
    row = manager.get_row(agent_id)
    user_id = row.user_id if row is not None else None
    if user_id is None:
        user_id = manager._connector_user_override.get(agent_id)
    runners_dict = manager.acp_settings.load_runners(user_id) if user_id is not None else {}
    return ACPConfig.from_dict({"runners": runners_dict})


def _parse_max_runtime(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


async def stream_acp_passthrough(
    manager: Any,
    agent_id: str,
    thread_id: str,
    request: dict[str, Any],
    passthrough_cfg: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Stream one passthrough turn as Octop chunks (never raises)."""
    runner = str(passthrough_cfg.get("runner") or "").strip()
    logger.info("acp passthrough hit: agent=%s runner=%s", agent_id, runner)

    acp_config = _build_acp_config(manager, agent_id)
    runner_cfg = acp_config.runners.get(runner)
    if runner_cfg is None or not runner_cfg.enabled:
        yield {
            "type": "error",
            "message": f"ACP passthrough runner '{runner}' is not configured or disabled.",
        }
        yield {"type": "done"}
        return

    workspace = manager.resolve_workspace_dir(agent_id)
    service_key = str(workspace)
    cwd_text = str(passthrough_cfg.get("cwd") or "").strip() or str(workspace)
    start_prompt = (
        str(passthrough_cfg.get("start_prompt") or "").strip() or _DEFAULT_START_PROMPT
    )
    max_runtime = _parse_max_runtime(
        passthrough_cfg.get("max_runtime", _DEFAULT_MAX_RUNTIME)
    )

    service = get_acp_service(service_key)
    if service is None:
        service = init_acp_service(service_key, acp_config)

    turn_text = extract_turn_text(request).strip()

    try:
        pending = await service.get_pending_permission(thread_id=thread_id, runner=runner)
    except Exception as exc:  # noqa: BLE001 — surfaced as a stream error chunk
        yield {"type": "error", "message": str(exc)}
        yield {"type": "done"}
        return

    queue: asyncio.Queue[Any] = asyncio.Queue()
    final_event: dict[str, Any] | None = None

    async def on_message(payload: dict[str, Any], _is_last: bool) -> None:
        nonlocal final_event
        text = render_event_text(payload)
        if text:
            await queue.put(text)
        if str(payload.get("type") or "").lower() == "text":
            final_event = payload

    async def run_turn() -> dict[str, Any]:
        if pending is not None:
            bound = await service.get_session(thread_id, runner)
            if bound is None:
                raise RuntimeError(
                    f"ACP session for runner '{runner}' not found while permission is pending"
                )
            return await service.resume_permission(
                acp_session_id=bound.acp_session_id,
                option_id=turn_text,
                on_message=on_message,
            )
        prompt_text = turn_text or start_prompt
        return await service.run_turn(
            thread_id=thread_id,
            runner=runner,
            prompt_blocks=[{"type": "text", "text": prompt_text}],
            cwd=cwd_text,
            on_message=on_message,
            restart=False,
            require_existing=False,
        )

    async def execute() -> dict[str, Any]:
        try:
            if max_runtime is None:
                return await run_turn()
            return await asyncio.wait_for(run_turn(), timeout=max_runtime)
        finally:
            await queue.put(_SENTINEL)

    task = asyncio.create_task(execute())
    emitted = 0
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            text = str(item)
            if text:
                emitted += 1
                yield {"type": "token", "content": text}
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        raise

    try:
        result = await task
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        with suppress(Exception):
            await service.cancel_turn(thread_id=thread_id, runner=runner)
        yield {
            "type": "error",
            "message": (
                f"ACP passthrough runner '{runner}' reached max_runtime and was interrupted."
            ),
        }
        yield {"type": "done"}
        return
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        yield {"type": "done"}
        return

    status = str(result.get("status") or "")
    if status == "permission_required":
        suspended = result.get("suspended_permission")
        permission_text = (
            format_permission_required(suspended)
            if suspended is not None
            else "Permission required. Reply with an option id."
        )
        if permission_text:
            yield {"type": "token", "content": permission_text}
    elif status == "completed":
        # Assistant text is accumulated by ACPHostedClient and returned as the
        # final event — it is *not* streamed through on_message. Forward it.
        event = result.get("event")
        if isinstance(event, dict) and str(event.get("type") or "").lower() == "text":
            text = str(event.get("text") or "").strip()
            if text:
                yield {"type": "token", "content": text}
        elif emitted == 0 and final_event is not None:
            text = str(final_event.get("text") or "").strip()
            if text:
                yield {"type": "token", "content": text}

    yield {"type": "done"}
