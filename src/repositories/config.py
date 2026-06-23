"""Connection settings read from the environment / a project-root ``.env``.

No ORM and no framework — just ``os.environ`` plus an optional ``python-dotenv``
load. Both discrete variables (``CLICKHOUSE_URL``, ``MYSQL_HOST``, ...) and a
single ``MYSQL_URI`` DSN are supported, mirroring the TWR core ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse, unquote


def _load_dotenv() -> None:
    """Best-effort ``.env`` load. Uses python-dotenv when available, otherwise
    parses the project-root ``.env`` by hand so the repos work without extra
    dependencies."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except Exception:
        pass

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ClickHouseSettings:
    host: str
    port: int
    username: str
    password: str
    database: str
    secure: bool

    @classmethod
    def from_env(cls) -> "ClickHouseSettings":
        # Prefer an explicit URL (http://host:8123), like the TWR core apps.
        url = os.getenv("CLICKHOUSE_URL")
        host = os.getenv("CLICKHOUSE_HOST", "localhost")
        port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        secure = os.getenv("CLICKHOUSE_SECURE", "false").lower() in ("1", "true", "yes")
        if url:
            parsed = urlparse(url)
            host = parsed.hostname or host
            port = parsed.port or port
            secure = parsed.scheme == "https"
        return cls(
            host=host,
            port=port,
            username=os.getenv("CLICKHOUSE_USERNAME", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            database=os.getenv("CLICKHOUSE_DATABASE", "shield"),
            secure=secure,
        )


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "MySQLSettings":
        # A single DSN (mysql://user:pass@host:3306/db) wins when present.
        uri = os.getenv("MYSQL_URI") or os.getenv("MYSQL_URL")
        if uri:
            parsed = urlparse(uri)
            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=unquote(parsed.username or "root"),
                password=unquote(parsed.password or ""),
                database=(parsed.path or "/").lstrip("/") or "local_TWR",
            )
        return cls(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "local_TWR"),
        )


@dataclass(frozen=True)
class Settings:
    clickhouse: ClickHouseSettings
    mysql: MySQLSettings


def get_settings() -> Settings:
    """Load ``.env`` (once) and return the resolved connection settings."""
    _load_dotenv()
    return Settings(
        clickhouse=ClickHouseSettings.from_env(),
        mysql=MySQLSettings.from_env(),
    )
