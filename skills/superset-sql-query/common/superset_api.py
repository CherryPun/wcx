#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://superset.yzh-logverse.k8s.qiniu.io"
DEFAULT_DATABASE_ID = 2
DEFAULT_SCHEMA = "jarvis"
DEFAULT_PROVIDER = "ldap"
DEFAULT_TIMEOUT = 60
DEFAULT_QUERY_LIMIT = 100000
LOCAL_SUPERSET_ENV = Path.home() / ".codex" / ".superset_env"
ALLOWED_ENV_KEYS = {"SUPERSET_USERNAME", "SUPERSET_PASSWORD"}


def load_local_superset_env(env_path: Path = LOCAL_SUPERSET_ENV) -> None:
    """Load shared Superset credentials from a local file when env vars are absent."""
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_ENV_KEYS:
            continue
        cleaned_value = value.strip().strip('"').strip("'")
        if cleaned_value:
            os.environ.setdefault(key, cleaned_value)


def env_or_die(name: str) -> str:
    load_local_superset_env()
    value = os.getenv(name)
    if value:
        return value
    raise SystemExit(f"Missing required environment variable: {name}")


def raise_for_status_with_body(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000]
        raise RuntimeError(f"{action} failed: {exc}\n{body}") from exc


class SupersetSQLClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        database_id: int = DEFAULT_DATABASE_ID,
        schema: str = DEFAULT_SCHEMA,
        provider: str = DEFAULT_PROVIDER,
        query_limit: int = DEFAULT_QUERY_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        username: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.database_id = database_id
        self.schema = schema
        self.provider = provider
        self.query_limit = query_limit
        self.timeout = timeout
        self.username = username
        self.password = password
        self.session = session or requests.Session()
        self.headers: dict[str, str] | None = None

    def login(self, force: bool = False) -> dict[str, str]:
        if self.headers and not force:
            return self.headers

        response = self.session.post(
            f"{self.base_url}/api/v1/security/login",
            json={
                "username": self.username or env_or_die("SUPERSET_USERNAME"),
                "password": self.password or env_or_die("SUPERSET_PASSWORD"),
                "provider": self.provider,
                "refresh": True,
            },
            timeout=self.timeout,
        )
        raise_for_status_with_body(response, "Superset login")
        access_token = response.json()["access_token"]

        headers = {"Authorization": f"Bearer {access_token}"}
        csrf_response = self.session.get(
            f"{self.base_url}/api/v1/security/csrf_token/",
            headers=headers,
            timeout=self.timeout,
        )
        raise_for_status_with_body(csrf_response, "Fetch CSRF token")

        headers["X-CSRFToken"] = csrf_response.json()["result"]
        headers["Content-Type"] = "application/json"
        self.headers = headers
        return headers

    def execute_sql(
        self,
        sql: str,
        *,
        schema: str | None = None,
        query_limit: int | None = None,
        tab: str = "Superset SQL",
        sql_editor_id: str = "script",
    ) -> list[dict[str, Any]]:
        payload = {
            "client_id": uuid.uuid4().hex[:10],
            "database_id": self.database_id,
            "json": True,
            "runAsync": False,
            "schema": schema or self.schema,
            "sql": sql,
            "sql_editor_id": sql_editor_id,
            "tab": tab,
            "tmp_table_name": "",
            "select_as_cta": False,
            "ctas_method": "TABLE",
            "templateParams": None,
            "queryLimit": query_limit or self.query_limit,
            "expand_data": True,
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/sqllab/execute/",
            headers=self.login(),
            json=payload,
            timeout=max(self.timeout, 180),
        )
        raise_for_status_with_body(response, "Execute SQL")
        body = response.json()
        if "data" not in body:
            raise RuntimeError(f"Unexpected SQL response: {json.dumps(body, ensure_ascii=False)[:2000]}")
        return body["data"]
