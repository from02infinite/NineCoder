from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ninecoder.errors import PermissionDenied, ToolError
from ninecoder.permissions import hits_sensitive_path
from ninecoder.sandbox import NullSandbox, SandboxBackend


MAX_READ_LINES = 400
MAX_SEARCH_MATCHES = 120
MAX_OUTPUT_CHARS = 12000
LINE_PREVIEW_CHARS = 2000

# Best-effort blacklist for `run_shell`. This is a tripwire against accidental
# damage, NOT a sandbox: commands still run as the current user and can read or
# modify files outside the workspace. Prefer `--permission ask` (or an OS-level
# sandbox) when the model handles untrusted tasks. The list is intentionally
# conservative on the *blocking* side — a false positive merely denies a command
# the model can rephrase; a false negative is a hole.
_BLOCKED_COMMANDS = {
    # Interactive editors and pagers would hang the non-interactive loop.
    "vim", "vi", "nano", "emacs", "less", "more", "top", "htop",
    # Privilege escalation.
    "sudo", "su", "doas", "pkexec", "sudoedit",
    # Destructive or administrative.
    "shutdown", "reboot", "halt", "poweroff", "telinit",
    "mkfs", "mkswap", "dd", "fsck", "fdisk", "parted", "partprobe",
}

# Neutral wrappers that don't change which command actually runs. (`sudo` and
# friends are NOT here — they are blocked themselves.)
_PREFIX_WORDS = {"env", "command", "builtin", "exec", "nice", "nohup", "time"}

_BLOCKED_PATTERNS = [
    # A tail/less following a stream never terminates.
    re.compile(r"\btail\b\s+(-\w*f\w*|-f|--follow)(\s|$)"),
    # Classic fork bomb.
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"),
]


def _is_blocked_command(command: str) -> bool:
    """Return True when a shell command looks too dangerous to run blindly."""
    normalized = command.lower()
    # Drop backslash escapes (`\sudo`) so the real command name is visible.
    normalized = re.sub(r"\\", "", normalized)
    tokens = normalized.split()
    if not tokens:
        return False
    name = _command_name(tokens)
    if name in _BLOCKED_COMMANDS:
        return True
    if name == "rm" and _is_dangerous_rm(tokens):
        return True
    body = " ".join(tokens)
    return any(pattern.search(body) for pattern in _BLOCKED_PATTERNS)


def _command_name(tokens: list[str]) -> str:
    """Basename of the first token that is neither an env assignment nor a
    neutral wrapper, e.g. ``env FOO=1 /usr/bin/sudo`` -> ``sudo``. ``.stem``
    drops a command-family suffix so ``mkfs.ext4`` matches ``mkfs``."""
    for token in tokens:
        if "=" in token and not token.startswith(("-", "--")):
            continue
        if token in _PREFIX_WORDS:
            continue
        return Path(token).stem
    return ""


def _is_dangerous_rm(tokens: list[str]) -> bool:
    """True for ``rm`` with recursive + force flags targeting a root-ish path."""
    flags = "".join(token for token in tokens if token.startswith("-") and not token.startswith("--"))
    recursive = bool(re.search(r"[rR]", flags) or "-R" in tokens or "--recursive" in tokens)
    force = bool(re.search(r"[fF]", flags) or "--force" in tokens)
    if not (recursive and force):
        return False
    targets = [token for token in tokens if not token.startswith("-")]
    return any(
        target.startswith(("/", "~", "..")) or target in {"*", "."}
        for target in targets
    )


@dataclass(frozen=True)
class CommandResult:
    output: str
    returncode: int
    timed_out: bool = False


class Workspace:
    def __init__(
        self,
        root: str | Path,
        *,
        sandbox: SandboxBackend | None = None,
        allow_network: bool = False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox or NullSandbox()
        self.allow_network = allow_network

    def resolve(self, path: str | Path = ".") -> Path:
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionDenied(f"path escapes workspace: {path}")
        if hits_sensitive_path(resolved):
            raise PermissionDenied(f"sensitive path is protected: {path}")
        return resolved

    def rel(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve(strict=False).relative_to(self.root))
        except ValueError:
            return str(path)

    def list_files(self, path: str = ".", limit: int = 200) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"path not found: {path}")
        if target.is_file():
            return self.rel(target)
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
            if child.name in {".git", "__pycache__", ".pytest_cache"}:
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{self.rel(child)}{suffix}")
            if len(entries) >= limit:
                entries.append(f"... ({limit} entries shown)")
                break
        return "\n".join(entries) or "(empty directory)"

    def read_file(self, path: str, offset: int = 1, limit: int = 200) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"file not found: {path}")
        if target.is_dir():
            return self.list_files(path)
        data = target.read_text(encoding="utf-8", errors="replace")
        lines = data.splitlines()
        offset = max(offset, 1)
        limit = max(1, min(limit, MAX_READ_LINES))
        start = offset - 1
        if start >= len(lines):
            raise ToolError(f"offset {offset} is beyond end of file ({len(lines)} lines)")
        selected = lines[start : start + limit]
        rendered = [
            f"{start + index + 1:>5}: {line[:LINE_PREVIEW_CHARS]}"
            + (" ... [line truncated]" if len(line) > LINE_PREVIEW_CHARS else "")
            for index, line in enumerate(selected)
        ]
        if start + limit < len(lines):
            rendered.append(f"... ({len(lines) - start - limit} more lines)")
        return "\n".join(rendered)

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return self._diff(path, old, content) or f"wrote {len(content)} bytes to {path}"

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        if old_text == new_text:
            raise ToolError("old_text and new_text are identical")
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"file not found: {path}")
        content = target.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count == 0:
            hint = self._closest_hint(content, old_text)
            raise ToolError(f"old_text did not match {path}.{hint}")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_text matched {count} times in {path}; set replace_all=true or narrow the text"
            )
        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        target.write_text(updated, encoding="utf-8")
        return self._diff(path, content, updated)

    def search(self, pattern: str, path: str = ".", include: str = "") -> str:
        # ``rg`` is deliberately not sandboxed: it is a fixed read-only binary
        # invoked with a validated pattern/path, so the risk surface is low and
        # sandboxing it would add failure modes for no real gain.
        target = self.resolve(path)
        if shutil.which("rg"):
            args = ["rg", "--line-number", "--no-heading", "--color", "never"]
            if include:
                args.extend(["--glob", include])
            args.extend([pattern, str(target)])
            result = subprocess.run(
                args,
                cwd=self.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )
            output = result.stdout.strip()
            return self._truncate(output or "(no matches)")
        return self._search_fallback(pattern, target)

    def run_shell(self, command: str, timeout: int = 30) -> CommandResult:
        command = command.strip()
        if not command:
            raise ToolError("command must not be empty")
        if _is_blocked_command(command):
            raise PermissionDenied(f"blocked shell command: {command}")
        env = os.environ | {"PAGER": "cat", "GIT_PAGER": "cat", "LESS": "-R"}
        if self.sandbox.available():
            wrapped = self.sandbox.wrap(
                command, self.root, allow_network=self.allow_network
            )
            proc = subprocess.Popen(
                wrapped.argv,
                cwd=self.root,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
            )
        else:
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
            )
        try:
            stdout, _ = proc.communicate(timeout=max(1, min(timeout, 300)))
            return CommandResult(self._truncate(stdout), proc.returncode)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, 9)
            else:
                proc.kill()
            stdout, _ = proc.communicate()
            return CommandResult(
                self._truncate(stdout) + f"\n[command timed out after {timeout}s]",
                -1,
                timed_out=True,
            )

    def _search_fallback(self, pattern: str, target: Path) -> str:
        matches: list[str] = []
        files = [target] if target.is_file() else target.rglob("*")
        for file in files:
            if not file.is_file() or file.name == ".git":
                continue
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if pattern in line:
                    matches.append(f"{self.rel(file)}:{number}:{line}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        matches.append(f"... ({MAX_SEARCH_MATCHES} matches shown)")
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def _diff(self, path: str, old: str, new: str) -> str:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )

    def _closest_hint(self, content: str, old_text: str) -> str:
        probe = next((line.strip() for line in old_text.splitlines() if line.strip()), "")
        if not probe:
            return ""
        candidates = content.splitlines()
        close = difflib.get_close_matches(probe, candidates, n=3, cutoff=0.45)
        if not close:
            return ""
        return "\nClosest lines:\n" + "\n".join(close)

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        head = text[: MAX_OUTPUT_CHARS // 2]
        tail = text[-MAX_OUTPUT_CHARS // 2 :]
        return f"{head}\n\n[... {len(text) - len(head) - len(tail)} chars omitted ...]\n\n{tail}"
