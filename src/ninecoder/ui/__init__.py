"""UI backends for NineCoder.

The agent core emits events through :class:`~ninecoder.ui.base.AgentUI`; this
package provides the concrete backends and a small factory to pick one.
"""
from __future__ import annotations

from ninecoder.ui.base import AgentUI, NullUI, UiContext
from ninecoder.ui.plain_ui import PlainUI

__all__ = ["AgentUI", "NullUI", "UiContext", "PlainUI", "make_ui"]


def make_ui(*, mode: str, context: UiContext) -> AgentUI:
    """Build a UI backend.

    ``mode`` is one of ``"null"``, ``"plain"``, or ``"rich"``. ``rich`` is
    imported lazily so plain/JSON modes never require ``rich``/``prompt_toolkit``.
    """
    if mode == "null":
        return NullUI(context)
    if mode == "plain":
        return PlainUI(context)
    if mode == "rich":
        try:
            from ninecoder.ui.rich_ui import RichUI
        except ImportError as exc:  # pragma: no cover - rich is a declared dep
            raise RuntimeError(
                "rich mode requires the 'rich' and 'prompt_toolkit' packages; "
                "run `pip install rich prompt_toolkit` or use --plain."
            ) from exc
        return RichUI(context)
    raise ValueError(f"unknown UI mode: {mode}")
