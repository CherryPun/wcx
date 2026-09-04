#!/usr/bin/env python3
"""Fetch and sanitize NiuLink virtual-to-real business bindings.

The Authorization value is read from NIULINK_AUTH and is never written to an
output file. Outputs contain only business identifiers, names, and binding
roles needed by the training pipeline.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://linkcloud-admin.qiniu.io"
VIRTUAL_CUSTOMERS_PATH = "/api/proxy/jarvis/v1/customers/virtual"
CUSTOMERS_PATH = "/api/proxy/jarvis/v1/customers"
MODE_NAMES = {0: "normal", 1: "accept", 2: "price"}


def request_json(
    base_url: str,
    path: str,
    params: dict[str, Any],
    authorization: str,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}?{query}",
        headers={"Authorization": authorization, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_all(
    base_url: str,
    path: str,
    authorization: str,
    page_size: int = 999,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = request_json(
            base_url,
            path,
            {
                "page": page,
                "size": page_size,
                "sortKey": "createAt" if path == VIRTUAL_CUSTOMERS_PATH else "name",
                "sortType": "desc" if path == VIRTUAL_CUSTOMERS_PATH else "asc",
                "permissionControl": "false",
                "includeChildSignatory": "false",
            },
            authorization,
        )
        batch = payload.get("items") or []
        items.extend(batch)
        total = int(payload.get("count") or len(items))
        if not batch or len(items) >= total:
            return items
        page += 1


def flatten_bindings(
    virtual_customers: list[dict[str, Any]],
    customers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    customer_names = {
        str(customer.get("id", "")): str(customer.get("name", "")).strip()
        for customer in customers
        if customer.get("id") is not None
    }
    rows: list[dict[str, Any]] = []
    sanitized_virtuals: list[dict[str, Any]] = []
    for virtual in virtual_customers:
        virtual_id = str(virtual.get("id", ""))
        associations: list[dict[str, Any]] = []
        for association in virtual.get("associatedCustomers") or []:
            real_id = str(association.get("customerId", ""))
            mode = int(association.get("mode", 0))
            item = {
                "business": real_id,
                "business_name": customer_names.get(real_id, ""),
                "binding_mode": mode,
                "binding_mode_name": MODE_NAMES.get(mode, "unknown"),
            }
            associations.append(item)
            rows.append({
                "virtual_business": virtual_id,
                "virtual_business_name": str(virtual.get("name", "")).strip(),
                "virtual_sign_id": str(virtual.get("signId", "")),
                "virtual_sign_name": str(virtual.get("signName", "")).strip(),
                **item,
            })
        sanitized_virtuals.append({
            "virtual_business": virtual_id,
            "virtual_business_name": str(virtual.get("name", "")).strip(),
            "external_name": str(virtual.get("externalName", "")).strip(),
            "virtual_sign_id": str(virtual.get("signId", "")),
            "virtual_sign_name": str(virtual.get("signName", "")).strip(),
            "associated_businesses": associations,
        })
    snapshot = {
        "source": f"{BASE_URL}/customer/manage/virtual?sortKey=createAt&sortType=desc",
        "virtual_business_count": len(sanitized_virtuals),
        "binding_count": len(rows),
        "virtual_businesses": sanitized_virtuals,
    }
    return rows, snapshot


def write_outputs(
    rows: list[dict[str, Any]],
    snapshot: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "virtual_business",
        "virtual_business_name",
        "virtual_sign_id",
        "virtual_sign_name",
        "business",
        "business_name",
        "binding_mode",
        "binding_mode_name",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    today = dt.date.today().strftime("%Y%m%d")
    default_dir = Path(__file__).resolve().parent / "business_binding_audit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_dir / f"niulink_virtual_business_bindings_{today}.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_dir / f"niulink_virtual_business_bindings_{today}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization = os.environ.get("NIULINK_AUTH", "").strip()
    if not authorization:
        raise SystemExit("NIULINK_AUTH is required")
    virtual_customers = fetch_all(args.base_url, VIRTUAL_CUSTOMERS_PATH, authorization)
    customers = fetch_all(args.base_url, CUSTOMERS_PATH, authorization)
    rows, snapshot = flatten_bindings(virtual_customers, customers)
    snapshot["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_outputs(rows, snapshot, args.output_csv, args.output_json)
    print(json.dumps({
        "virtual_businesses": len(virtual_customers),
        "bindings": len(rows),
        "output_csv": str(args.output_csv),
        "output_json": str(args.output_json),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
