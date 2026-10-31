import asyncio
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch
from urllib.request import Request

from action_client import ActionApiClient, ActionApiError


class ActionApiClientTests(unittest.TestCase):
    def test_client_reads_action_token_from_project_env_name(self) -> None:
        with patch.dict(os.environ, {"ai_top_100_api_key": "test-token"}):
            client = ActionApiClient()

        self.assertEqual(client._token, "test-token")

    def test_request_sets_auth_and_test_case_headers(self) -> None:
        client = ActionApiClient(token="test-token")

        with patch("action_client.urlopen") as urlopen:
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'{"ok":true}'
            urlopen.return_value = response

            result = asyncio.run(
                client.request(
                    "POST",
                    "/api/escalations",
                    "L-32",
                    {"lane": "LANE2_HANDOFF", "reason": "review"},
                )
            )

        sent_request = urlopen.call_args.args[0]
        self.assertIsInstance(sent_request, Request)
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(
            sent_request.get_header("X-aitop100-test-case-id"), "L-32"
        )
        self.assertEqual(
            json.loads(sent_request.data),
            {"lane": "LANE2_HANDOFF", "reason": "review"},
        )
        self.assertEqual(result, {"ok": True})

    def test_request_requires_valid_test_case_id(self) -> None:
        client = ActionApiClient(token="test-token")

        with self.assertRaises(ValueError):
            asyncio.run(client.request("GET", "/api/orders/123", ""))

    def test_request_rejects_external_urls(self) -> None:
        client = ActionApiClient(token="test-token")

        with self.assertRaises(ValueError):
            asyncio.run(
                client.request("GET", "https://example.com/api/orders", "L-32")
            )

    def test_rate_limit_exposes_retry_after_without_response_body(self) -> None:
        client = ActionApiClient(token="test-token")
        http_error = HTTPError(
            "https://gyodong-mall-action.aitop100.org/api/orders/123",
            429,
            "Too Many Requests",
            {"Retry-After": "5"},
            None,
        )

        with patch("action_client.urlopen", side_effect=http_error):
            with self.assertRaises(ActionApiError) as raised:
                asyncio.run(client.request("GET", "/api/orders/123", "L-32"))

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.retry_after, "5")


if __name__ == "__main__":
    unittest.main()
