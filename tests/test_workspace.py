import tempfile
import unittest
from pathlib import Path

from ninecoder.errors import PermissionDenied, ToolError
from ninecoder.permissions import hits_sensitive_path
from ninecoder.workspace import Workspace, _is_blocked_command


class ShellBlocklistTest(unittest.TestCase):
    def assert_blocked(self, command: str) -> None:
        self.assertTrue(_is_blocked_command(command), f"should be blocked: {command}")

    def assert_allowed(self, command: str) -> None:
        self.assertFalse(_is_blocked_command(command), f"should be allowed: {command}")

    def test_blocks_privilege_escalation(self) -> None:
        for command in [
            "sudo rm -rf /",
            "sudo echo hi",
            "env sudo whoami",
            "FOO=1 sudo whoami",
            "\\sudo whoami",
            "/usr/bin/sudo whoami",
            "su",
            "doas bash",
            "pkexec ls",
        ]:
            self.assert_blocked(command)

    def test_blocks_interactive_commands(self) -> None:
        for command in [
            "vim",
            "nano file.txt",
            "emacs -nw",
            "less README.md",
            "more file.txt",
            "top",
            "htop",
            "tail -f app.log",
            "tail --follow app.log",
        ]:
            self.assert_blocked(command)

    def test_blocks_destructive_commands(self) -> None:
        for command in [
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "fdisk /dev/sda",
            "mkswap /dev/sda1",
        ]:
            self.assert_blocked(command)

    def test_blocks_dangerous_rm_targets(self) -> None:
        for command in [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf ~",
            "rm -rf ..",
            "rm -rf ../..",
            "rm -rf *",
            "rm -r -f /tmp",
            "rm -fr /home/user",
            "rm --recursive --force /",
        ]:
            self.assert_blocked(command)

    def test_allows_workspace_relative_rm(self) -> None:
        for command in [
            "rm -rf build",
            "rm -rf node_modules",
            "rm -f build/a.o",
            "rm build/a.o",
        ]:
            self.assert_allowed(command)

    def test_allows_common_dev_commands(self) -> None:
        for command in [
            "python -m unittest -q",
            "pytest",
            "git status",
            "git commit -m 'fix bug'",
            "echo hello",
            "ls -la",
            "cat README.md",
            "mkdir -p build",
        ]:
            self.assert_allowed(command)

    def test_blocks_fork_bomb(self) -> None:
        self.assert_blocked(":(){ :|:& };:")


class WorkspaceTest(unittest.TestCase):
    def test_read_file_uses_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            workspace = Workspace(root)

            self.assertIn("2: two", workspace.read_file("a.txt", offset=2, limit=1))

    def test_edit_file_returns_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('bad')\n", encoding="utf-8")
            workspace = Workspace(root)

            diff = workspace.edit_file("a.py", "bad", "good")

            self.assertIn("+print('good')", diff)

    def test_edit_file_reports_missing_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('hello')\n", encoding="utf-8")
            workspace = Workspace(root)

            with self.assertRaises(ToolError):
                workspace.edit_file("a.py", "missing", "value")

    def test_paths_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)

            with self.assertRaises(PermissionDenied):
                workspace.resolve("../outside.txt")


class SensitivePathTest(unittest.TestCase):
    def test_detects_dotenv_variants(self) -> None:
        for path in [".env", ".env.local", ".env.production", ".env.backup"]:
            self.assertTrue(hits_sensitive_path(path), path)

    def test_detects_private_keys(self) -> None:
        for path in ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "~/.ssh/id_rsa"]:
            self.assertTrue(hits_sensitive_path(path), path)

    def test_detects_sensitive_directories(self) -> None:
        for path in [".ssh/config", ".aws/credentials", ".kube/config", "~/.gnupg/secring.gpg"]:
            self.assertTrue(hits_sensitive_path(path), path)

    def test_detects_sensitive_suffixes(self) -> None:
        for path in ["server.pem", "id_rsa.key", "cert.p12", "keystore.pfx"]:
            self.assertTrue(hits_sensitive_path(path), path)

    def test_allows_ordinary_files(self) -> None:
        for path in ["app.py", "README.md", ".gitignore", "config.example.json"]:
            self.assertFalse(hits_sensitive_path(path), path)


if __name__ == "__main__":
    unittest.main()
