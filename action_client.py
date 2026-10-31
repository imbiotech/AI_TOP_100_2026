from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_BASE_URL = "https://gyodong-mall-action.aitop100.org"
_TEST_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class ActionApiError(RuntimeError):
    """Raised when an Action API request cannot be completed successfully."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ActionApiClient:
    """Call the contest Action API without exposing credentials in errors."""

    def __init__(self, token: str | None = None) -> None:
        self._token = (
            token
            or os.getenv("ai_top_100_api_key")
            or os.getenv("AITOP100_ACTION_TOKEN")
        )
        if not self._token:
            raise ValueError("The Action API token is not configured.")

    async def request(
        self,
        method: str,
        path: str,
        test_case_id: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request with the required test-case header."""
        normalized_method = method.upper()
        parsed_path = urlsplit(path)
        if (
            normalized_method not in {"GET", "POST"}
            or not path.startswith("/api/")
            or parsed_path.netloc
            or parsed_path.scheme
        ):
            raise ValueError("Only relative Action API paths are supported.")
        if not _TEST_CASE_ID_PATTERN.fullmatch(test_case_id):
            raise ValueError("A valid test case ID is required.")

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{_BASE_URL}{path}",
            data=body,
            method=normalized_method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Aitop100-Test-Case-Id": test_case_id,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        return await asyncio.to_thread(self._send, request)

    @staticmethod
    def _send(request: Request) -> Any:
        try:
            with urlopen(request, timeout=8) as response:
                response_body = response.read()
        except HTTPError as error:
            raise ActionApiError(
                f"Action API returned HTTP {error.code}.",
                status_code=error.code,
                retry_after=error.headers.get("Retry-After"),
            ) from None
        except (URLError, TimeoutError) as error:
            raise ActionApiError(
                "Action API request failed or timed out."
            ) from error

        if not response_body:
            return None
        try:
            return json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ActionApiError(
                "Action API returned an invalid JSON response."
            ) from error
