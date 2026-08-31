import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ninecoder.sandbox import (
    BwrapSandbox,
    NullSandbox,
    SandboxExecSandbox,
    WrappedCommand,
    detect_sandbox,
    sandbox_description,
)
from ninecoder.workspace import Workspace


class DetectSandboxTest(unittest.TestCase):
    def test_off_selects_null(self) -> None:
        for selector in ["off", "none", ""]:
            self.assertIsInstance(detect_sandbox(selector), NullSandbox)

    def test_explicit_bwrap_forces_backend(self) -> None:
        backend = detect_sandbox("bwrap")
        self.assertIsInstance(backend, BwrapSandbox)

    def test_explicit_sandbox_exec_forces_backend(self) -> None:
        backend = detect_sandbox("sandbox-exec")
        self.assertIsInstance(backend, SandboxExecSandbox)

    def test_unknown_selector_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_sandbox("docker")

    def test_auto_returns_null_when_no_binary(self) -> None:
        with mock.patch("ninecoder.sandbox.shutil.which", return_value=None):
            self.assertIsInstance(detect_sandbox("auto"), NullSandbox)

    def test_auto_prefers_bwrap_over_sandbox_exec(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/bwrap" if name == "bwrap" else None

        with mock.patch("ninecoder.sandbox.shutil.which", side_effect=which):
            self.assertIsInstance(detect_sandbox("auto"), BwrapSandbox)


class BwrapArgvTest(unittest.TestCase):
    def test_wrap_blocks_network_by_default(self) -> None:
        argv = BwrapSandbox().wrap("echo hi", Path("/ws"), allow_network=False).argv
        self.assertIn("--ro-bind", argv)
        self.assertIn("/ws", argv)
        self.assertIn("--unshare-net", argv)
        self.assertEqual(argv[-4:], ["--", "bash", "-c", "echo hi"])

    def test_wrap_allows_network_when_requested(self) -> None:
        argv = BwrapSandbox().wrap("echo hi", Path("/ws"), allow_network=True).argv
        self.assertNotIn("--unshare-net", argv)

    def test_wrap_binds_workspace_writable(self) -> None:
        argv = BwrapSandbox().wrap("true", Path("/my/ws"), allow_network=False).argv
        self.assertTrue(_has_subsequence(argv, ["--bind", "/my/ws", "/my/ws"]))


class SandboxExecProfileTest(unittest.TestCase):
    def test_wrap_writes_profile_denying_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            wrapped = SandboxExecSandbox().wrap("echo hi", cwd, allow_network=False)
            argv = wrapped.argv
            self.assertEqual(argv[0], "sandbox-exec")
            self.assertEqual(argv[1], "-f")
            profile = Path(argv[2])
            self.assertTrue(profile.exists())
            content = profile.read_text(encoding="utf-8")
            self.assertIn("(deny default)", content)
            self.assertIn("(deny network*)", content)
            self.assertIn(str(cwd), content)

    def test_wrap_allows_network_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = SandboxExecSandbox().wrap("echo hi", Path(tmp), allow_network=True)
            content = Path(wrapped.argv[2]).read_text(encoding="utf-8")
            self.assertNotIn("(deny network*)", content)


class WorkspaceSandboxTest(unittest.TestCase):
    def test_run_shell_uses_wrapped_argv_when_available(self) -> None:
        class StubSandbox:
            name = "stub"

            def available(self) -> bool:
                return True

            def wrap(self, command: str, cwd: Path, *, allow_network: bool) -> WrappedCommand:
                return WrappedCommand(["stub", "run", command])

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp, sandbox=StubSandbox())
            with mock.patch("ninecoder.workspace.subprocess.Popen") as popen:
                popen.return_value.communicate.return_value = ("ok", None)
                popen.return_value.returncode = 0
                result = workspace.run_shell("echo hi")

            self.assertEqual(result.output, "ok")
            call_kwargs = popen.call_args.kwargs
            self.assertEqual(call_kwargs["shell"], False)
            self.assertEqual(popen.call_args.args[0], ["stub", "run", "echo hi"])

    def test_run_shell_skips_sandbox_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)  # NullSandbox
            with mock.patch("ninecoder.workspace.subprocess.Popen") as popen:
                popen.return_value.communicate.return_value = ("ok", None)
                popen.return_value.returncode = 0
                workspace.run_shell("echo hi")

            self.assertTrue(popen.call_args.kwargs["shell"])


class SandboxDescriptionTest(unittest.TestCase):
    def test_off_description(self) -> None:
        self.assertEqual(sandbox_description(NullSandbox(), False), "off")

    def test_bwrap_description_blocks_network(self) -> None:
        self.assertEqual(sandbox_description(BwrapSandbox(), False), "bwrap (network off)")

    def test_bwrap_description_allows_network(self) -> None:
        self.assertEqual(sandbox_description(BwrapSandbox(), True), "bwrap (network on)")


@unittest.skipUnless(
    shutil.which("bwrap") or shutil.which("sandbox-exec"),
    "no sandbox binary available",
)
class SandboxIntegrationTest(unittest.TestCase):
    def test_echo_runs_inside_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = detect_sandbox("auto")
            if isinstance(backend, NullSandbox):
                self.skipTest("auto resolved to no sandbox")
            workspace = Workspace(tmp, sandbox=backend, allow_network=True)
            result = workspace.run_shell("echo sandboxed")
            self.assertIn("sandboxed", result.output)


def _has_subsequence(argv: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return any(argv[i : i + n] == sub for i in range(len(argv) - n + 1))


if __name__ == "__main__":
    unittest.main()
