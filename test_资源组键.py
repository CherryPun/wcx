# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from 资源组键 import first_schedule_isp, group_key, ipv6_bin, quality_bin, transprov_bin


class ResourceGroupKeyTest(unittest.TestCase):
    def test_empty_schedule_is_local_isp(self) -> None:
        self.assertEqual(first_schedule_isp("", "电信"), "电信")
        self.assertEqual(first_schedule_isp("联通,移动", "电信"), "联通")

    def test_transprov_only_zero_or_hundred(self) -> None:
        self.assertEqual(transprov_bin(None), "本省")
        self.assertEqual(transprov_bin(0), "本省")
        self.assertEqual(transprov_bin(100), "出省")
        self.assertEqual(transprov_bin(50), "未知")

    def test_quality_has_liang_bin(self) -> None:
        self.assertEqual(quality_bin(96), "优")
        self.assertEqual(quality_bin(91), "良")
        self.assertEqual(quality_bin(85), "中")
        self.assertEqual(quality_bin(10), "差")
        self.assertEqual(quality_bin(None), "未知")

    def test_ipv6_and_full_key(self) -> None:
        self.assertEqual(ipv6_bin(True), "支持")
        key = group_key({
            "province": "上海", "city": "上海", "isp": "移动",
            "scheduleisps": "", "transprovrate": 100, "nattype": "fullcone",
            "ipv6": True, "quality_pct": 92,
        })
        self.assertEqual(key, "上海|上海|移动|移动|出省|full|支持|良")


if __name__ == "__main__":
    unittest.main()
