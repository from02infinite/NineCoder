"""Pluggable OS sandbox backends for :meth:`~ninecoder.workspace.Workspace.run_shell`.

NineCoder's shell tool runs commands as the current user. The command blocklist
in :mod:`ninecoder.workspace` is a tripwire, not a boundary: a model can still
write outside the workspace or exfiltrate data over the network. A sandbox
backend adds a real boundary by wrapping the command in an OS isolation layer.

A backend only *constructs an argv*; the actual process is still spawned by
:class:`~ninecoder.workspace.Workspace`. This keeps the surface small and easy
to test without needing the sandbox binary at import time.

Two backends are provided:

* :class:`BwrapSandbox` (Linux) — bubblewrap. Filesystem is read-only except the
  workspace and ``/tmp``, the process gets fresh namespaces, and network is
  denied by default. This is the primary, full-strength backend.
* :class:`SandboxExecSandbox` (macOS) — ``sandbox-exec``. Apple deprecated
  ``sandbox-exec`` (a thin wrapper over the private Seatbelt framework that may
  vanish in a future release) and its profile language is fiddly, so this is
  best-effort: filesystem writes are confined to the workspace and network is
  denied by default, but the isolation is weaker than bubblewrap.

Neither hides the home directory, so a sandboxed process can still *read*
``~/.ssh`` and friends — but with network denied by default it cannot send that
content anywhere. Treat the sandbox as filesystem + process + network isolation,
not a full secret container; sensitive-path denial (see
:mod:`ninecoder.permissions`) remains the defense for read access.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class WrappedCommand:
    """A ready-to-exec argv with ``shell=False``."""

    argv: list[str]


class SandboxBackend(Protocol):
    name: str

    def available(self) -> bool:
        """Whether the backend will actually isolate (binary present, or forced)."""

    def wrap(self, command: str, cwd: Path, *, allow_network: bool) -> WrappedCommand:
        """Wrap ``command`` so it runs inside the sandbox with ``cwd`` writable."""


class NullSandbox:
    """No sandbox. ``available`` is False so ``run_shell`` keeps its current
    ``shell=True`` path and behavior is byte-for-byte unchanged."""

    name = "off"

    def available(self) -> bool:
        return False

    def wrap(self, command: str, cwd: Path, *, allow_network: bool) -> WrappedCommand:
        raise RuntimeError("NullSandbox.wrap must not be called")


class BwrapSandbox:
    """bubblewrap (``bwrap``) isolation, the primary Linux backend."""

    name = "bwrap"

    def __init__(self, forced: bool = False) -> None:
        self._forced = forced

    def available(self) -> bool:
        return self._forced or shutil.which("bwrap") is not None

    def wrap(self, command: str, cwd: Path, *, allow_network: bool) -> WrappedCommand:
        argv = [
            "bwrap",
            # Read-only view of the whole filesystem, then re-mount the
            # workspace writable so edits are confined to it.
            "--ro-bind", "/", "/",
            "--bind", str(cwd), str(cwd),
            "--dev", "/dev",
            "--proc", "/proc",
            # Fresh writable tmp, otherwise programs that need /tmp break.
            "--tmpfs", "/tmp",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
            "--die-with-parent",
            "--new-session",
        ]
        if not allow_network:
            argv.append("--unshare-net")
        argv += ["--", "bash", "-c", command]
        return WrappedCommand(argv)


class SandboxExecSandbox:
    """``sandbox-exec`` (macOS), best-effort and deprecated by Apple."""

    name = "sandbox-exec"

    def __init__(self, forced: bool = False) -> None:
        self._forced = forced

    def available(self) -> bool:
        return self._forced or shutil.which("sandbox-exec") is not None

    def wrap(self, command: str, cwd: Path, *, allow_network: bool) -> WrappedCommand:
        path = self._profile_path()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self._profile(cwd, allow_network))
        return WrappedCommand(["sandbox-exec", "-f", path, "bash", "-c", command])

    def _profile(self, cwd: Path, allow_network: bool) -> str:
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            # Read access everywhere (home dirs included — see module note), but
            # writes only inside the workspace, the system temp dirs, and the
            # /dev pseudo-devices that ordinary redirects (2>/dev/null) rely on.
            "(allow file-read*)",
            "(allow file-read-metadata)",
            "(allow file-write*",
            f'    (subpath "{cwd}")',
            '    (subpath "/tmp")',
            '    (subpath "/private/tmp")',
            '    (literal "/dev/null")',
            '    (literal "/dev/zero")',
            '    (literal "/dev/urandom")',
            '    (literal "/dev/random")',
            '    (literal "/dev/tty")',
            '    (literal "/dev/stdout")',
            '    (literal "/dev/stderr")',
            '    (literal "/dev/stdin"))',
            "(allow file-write-create",
            f'    (subpath "{cwd}")',
            '    (subpath "/tmp")',
            '    (subpath "/private/tmp"))',
            "(allow file-ioctl)",
        ]
        if not allow_network:
            lines.append("(deny network*)")
        return "\n".join(lines) + "\n"

    def _profile_path(self) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".sb", prefix="ninecoder-", delete=False
        )
        handle.close()
        return handle.name


def detect_sandbox(mode: str) -> SandboxBackend:
    """Resolve a sandbox selector into a backend.

    ``mode`` is one of ``auto``, ``off``/``""``, ``bwrap``, or ``sandbox-exec``.
    ``auto`` picks the first backend whose binary is present; explicit names
    force that backend even when the binary is missing, so the first shell
    command fails loudly at spawn instead of silently running unsandboxed.
    """
    selector = (mode or "").strip().lower()
    if selector in {"", "off", "none"}:
        return NullSandbox()
    if selector == "auto":
        if BwrapSandbox().available():
            return BwrapSandbox()
        if SandboxExecSandbox().available():
            return SandboxExecSandbox()
        return NullSandbox()
    if selector == "bwrap":
        return BwrapSandbox(forced=True)
    if selector == "sandbox-exec":
        return SandboxExecSandbox(forced=True)
    raise ValueError(f"unknown sandbox backend: {mode}")


def sandbox_description(backend: SandboxBackend, allow_network: bool) -> str:
    """One-line summary for the startup banner."""
    if isinstance(backend, NullSandbox):
        return "off"
    network = "network on" if allow_network else "network off"
    return f"{backend.name} ({network})"
