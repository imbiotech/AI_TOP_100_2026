import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main


class KakaoWebhookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        main._CONVERSATIONS.clear()

    async def test_invalid_payload_returns_simple_text(self) -> None:
        request = SimpleNamespace(
            json=AsyncMock(return_value=[]),
            headers={},
        )

        response = await main.kakao_webhook(request)

        self.assertEqual(response["version"], "2.0")
        self.assertEqual(
            response["template"]["outputs"][0]["simpleText"]["text"],
            "문의 내용을 확인하지 못했습니다. 메시지를 다시 보내주세요.",
        )

    async def test_order_lookup_is_skipped_without_test_case_id(self) -> None:
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "userRequest": {
                        "utterance": "주문번호 100100003 환불 상태 알려주세요."
                    }
                }
            ),
            headers={},
        )

        with patch("main.ActionApiClient") as client_class:
            response = await main.kakao_webhook(request)

        client_class.assert_not_called()
        text = response["template"]["outputs"][0]["simpleText"]["text"]
        self.assertIn("현재 이 채널에서는 주문 상태를 확인할 수 없어요", text)
        self.assertNotIn("평가 식별정보", text)
        self.assertLessEqual(len(text), 500)

    async def test_order_response_does_not_disclose_personal_fields(self) -> None:
        client = SimpleNamespace(
            request=AsyncMock(
                return_value={
                    "order_no": "100100003",
                    "status": "PAID",
                    "recipient_name": "Test Customer",
                    "shipping_address": "Private address",
                }
            )
        )

        with patch("main.ActionApiClient", return_value=client):
            response = await main._handle_message(
                "주문번호 100100003 환불 상태 확인",
                "L-05",
                "kakao-user",
            )

        self.assertIn("결제 확인", response)
        self.assertNotIn("Test Customer", response)
        self.assertNotIn("Private address", response)

    async def test_explicit_health_claim_looks_up_then_records_escalation(self) -> None:
        client = SimpleNamespace(request=AsyncMock())
        client.request.side_effect = [
            {"order_no": "100100003", "status": "PAID"},
            [],
            {"order_no": "100100003", "status": "PAID"},
            [],
            {"escalation_id": "ESC-20371031-001", "request_status": "SUBMITTED"},
        ]

        with patch("main.ActionApiClient", return_value=client):
            first_response = await main._handle_message(
                "수박을 먹고 배탈이 난 것 같아요.", "L-32", "kakao-user"
            )
            order_response = await main._handle_message(
                "주문번호 100100003입니다.", "L-32", "kakao-user"
            )
            claim_response = await main._handle_message(
                "병원비 검토를 요청합니다.", "L-32", "kakao-user"
            )

        self.assertIn("주문번호", first_response)
        self.assertIn("관련성은 확인되지 않았어요", order_response)
        self.assertIn("ESC-20371031-001", claim_response)
        self.assertIn("영업일 3일", claim_response)
        self.assertEqual(client.request.await_count, 5)
        escalation_call = client.request.await_args_list[-1].args
        self.assertEqual(escalation_call[0], "POST")
        self.assertEqual(escalation_call[1], "/api/escalations")
        self.assertEqual(
            escalation_call[3]["lane"],
            "LANE3_CLAIM",
        )
        self.assertLessEqual(len(claim_response), 500)
        self.assertNotIn("kakao-user", main._CONVERSATIONS)

    async def test_webhook_uses_only_simple_text_output(self) -> None:
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "userRequest": {
                        "utterance": "안녕하세요",
                        "user": {"id": "kakao-user"},
                    }
                }
            ),
            headers={},
        )

        response = await main.kakao_webhook(request)

        outputs = response["template"]["outputs"]
        self.assertEqual(len(outputs), 1)
        self.assertIn("simpleText", outputs[0])


if __name__ == "__main__":
    unittest.main()
