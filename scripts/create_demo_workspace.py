from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "demo_workspace"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "calculator.py").write_text(
        """def divide(a, b):
    return a / b


def add(a, b):
    return a + b
""",
        encoding="utf-8",
    )
    (ROOT / "test_calculator.py").write_text(
        """import unittest

from calculator import add, divide


class CalculatorTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_divide(self):
        self.assertEqual(divide(6, 2), 3)

    def test_divide_by_zero_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "division by zero"):
            divide(1, 0)


if __name__ == "__main__":
    unittest.main()
""",
        encoding="utf-8",
    )
    print(ROOT)


if __name__ == "__main__":
    main()
