#!/usr/bin/env python3
from __future__ import annotations

import unittest

import fetch_niulink_virtual_business_bindings as bindings


class VirtualBusinessBindingTest(unittest.TestCase):
    def test_flatten_bindings_joins_real_business_names_and_modes(self) -> None:
        virtuals = [{
            "id": 100,
            "name": "虚拟业务",
            "signId": 10,
            "signName": "签约方",
            "associatedCustomers": [
                {"customerId": 200, "mode": 1},
                {"customerId": 201, "mode": 0},
            ],
        }]
        customers = [
            {"id": 200, "name": "真实业务A"},
            {"id": 201, "name": "真实业务B"},
        ]

        rows, snapshot = bindings.flatten_bindings(virtuals, customers)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["business_name"], "真实业务A")
        self.assertEqual(rows[0]["binding_mode_name"], "accept")
        self.assertEqual(rows[1]["binding_mode_name"], "normal")
        self.assertEqual(snapshot["virtual_business_count"], 1)
        self.assertEqual(snapshot["binding_count"], 2)


if __name__ == "__main__":
    unittest.main()
