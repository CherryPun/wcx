# -*- coding: utf-8 -*-
"""[wcx2 新增] 账本契约测试（新）：固定样本包 + 故意破坏的变体。

运行：python -m unittest test_ledger_contract.py
不依赖任何外部数据（fixtures/ 下自带样本）。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / "wcx2实验代码（新）" / "账本契约检查（新）.py"
FIXTURE = ROOT / "fixtures" / "ledger_sample_v1.csv"


def run_checker(path: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(CHECKER), str(path)] + (extra or [])
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


class LedgerContractTest(unittest.TestCase):
    def test_fixture_passes(self) -> None:
        r = run_checker(FIXTURE)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_nan_fails(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        broken = text.replace("2026-08-10,nodeB,10000280,1000", "2026-08-10,nodeB,10000280,")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nan.csv"
            p.write_text(broken, encoding="utf-8")
            r = run_checker(p)
        self.assertNotEqual(r.returncode, 0, "空值应被拦截")
        self.assertIn("空值", r.stdout)

    def test_duplicate_key_fails(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        line = "2026-08-10,nodeA,10000079,5000,250.0,300.0,1.0,1,true"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dup.csv"
            p.write_text(text + line + "\n", encoding="utf-8")
            r = run_checker(p)
        self.assertNotEqual(r.returncode, 0, "重复键应被拦截")
        self.assertIn("键唯一", r.stdout)

    def test_nonpositive_bandwidth_fails(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        broken = text.replace("2026-08-10,nodeB,10000280,1000", "2026-08-10,nodeB,10000280,0")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bw0.csv"
            p.write_text(broken, encoding="utf-8")
            r = run_checker(p)
        self.assertNotEqual(r.returncode, 0, "带宽<=0 应被拦截")

    def test_multi_business_flag_is_reported(self) -> None:
        r = run_checker(FIXTURE)
        self.assertIn("多业务行=2", r.stdout)

    def test_business_not_in_allowlist_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            al = Path(tmp) / "allow.csv"
            al.write_text("business,business_name\n10000079,x\n", encoding="utf-8")
            r = run_checker(FIXTURE, ["--allowlist", str(al)])
        self.assertNotEqual(r.returncode, 0, "越界业务应被拦截")
        self.assertIn("白名单", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
