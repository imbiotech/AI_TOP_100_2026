import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main
from ai_interpreter import Interpretation, interpret_message

class KakaoWebhookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        main._CONVERSATIONS.clear()
        agent_patcher = patch("main.agent.respond", AsyncMock(return_value=None))
        agent_patcher.start()
        self.addCleanup(agent_patcher.stop)
        patcher = patch("main.interpret_message", AsyncMock(return_value=None))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_ai_interpretation_routes_free_form_message(self) -> None:
        interpretation = Interpretation("return", None, True)
        main._remember_conversation("u2", "return", None)
        with patch(
            "main.interpret_message", AsyncMock(return_value=interpretation)
        ):
            text = await main._handle_message(
                "모른다니까 집 앞에 왔는데 안 먹고 싶어", None, "u2"
            )
        self.assertIn("모르셔도 괜찮아요", text)

    async def test_ai_unknown_flag_ignored_on_first_message(self) -> None:
        with patch(
            "main.interpret_message",
            AsyncMock(return_value=Interpretation("return", None, True)),
        ):
            text = await main._handle_message("반품해죠", None, "u4")
        self.assertIn("반품 가능 여부", text)

    async def test_ai_claim_does_not_escalate_without_explicit_request(self) -> None:
        interpretation = Interpretation("claim", "100100003", False)
        client = SimpleNamespace(request=AsyncMock(side_effect=[
            {"order_no": "100100003", "status": "PAID"}, [],
        ]))
        with patch(
            "main.interpret_message", AsyncMock(return_value=interpretation)
        ), patch("main.ActionApiClient", return_value=client):
            await main._handle_message("100100003 먹고 속이 안 좋아", "L-32", "u3")
        methods = [call.args[0] for call in client.request.await_args_list]
        self.assertNotIn("POST", methods)

    async def test_interpreter_rejects_invented_order_number(self) -> None:
        with patch(
            "ai_interpreter._call",
            return_value={"intent": "order", "order_number": "999999999"},
        ):
            result = await interpret_message("주문 확인해줘")
        self.assertIsNotNone(result)
        self.assertIsNone(result.order_number)

    async def test_interpreter_rejects_unknown_intent(self) -> None:
        with patch("ai_interpreter._call", return_value={"intent": "refund_now"}):
            self.assertIsNone(await interpret_message("환불해줘"))

    async def test_test_case_id_is_read_from_utterance_and_kept(self) -> None:
        main._TEST_CASE_IDS.clear()
        seen: list[str | None] = []

        async def fake_handle(text, test_case_id, user_id):
            seen.append(test_case_id)
            return "ok"

        def make(utterance: str) -> SimpleNamespace:
            return SimpleNamespace(
                json=AsyncMock(
                    return_value={
                        "userRequest": {
                            "utterance": utterance,
                            "user": {"id": "tc-user"},
                        }
                    }
                ),
                headers={},
            )

        with patch("main._handle_message", fake_handle):
            await main.kakao_webhook(make("[TC: L-32] 주문번호 100100003"))
            await main.kakao_webhook(make("병원 다녀왔어요"))
            await main.kakao_webhook(make("상담 정보 id: HX-9-z 인입 계정: 홍길동 안녕"))
            await main.kakao_webhook(make("다음 질문"))
        self.assertEqual(seen, ["L-32", "L-32", "HX-9-z", "HX-9-z"])
        main._TEST_CASE_IDS.clear()

    async def test_address_change_flow_l03(self) -> None:
        main._ADDRESS_FLOWS.clear()
        client = SimpleNamespace(request=AsyncMock(side_effect=[
            [], {"request_id": "ACR-1", "request_status": "SUBMITTED"},
        ]))
        with patch("main.ActionApiClient", return_value=client):
            t1 = await main._handle_message(
                "주소 잘못 넣었어요. 경기 성남시로 바꿔주세요.", "L-03", "a1")
            t2 = await main._handle_message("123456 이야", "L-03", "a1")
            t3 = await main._handle_message("네 맞아요", "L-03", "a1")
            t4 = await main._handle_message(
                "경기 성남시 분당구 판교로 12, 101동 202호", "L-03", "a1")
        self.assertIn("주문번호", t1)
        self.assertIn("대상이 맞나요", t2)
        self.assertIn("상세주소", t3)
        self.assertIn("ACR-1", t4)
        calls = client.request.await_args_list
        self.assertEqual(calls[0].args[:2], ("GET", "/api/shipments?order_no=123456"))
        self.assertEqual(
            calls[1].args[:2],
            ("POST", "/api/orders/123456/address-change-requests"),
        )
        self.assertEqual(
            calls[1].args[3],
            {"new_address": "경기 성남시 분당구 판교로 12, 101동 202호"},
        )

    async def test_address_change_blocked_when_shipped(self) -> None:
        main._ADDRESS_FLOWS.clear()
        client = SimpleNamespace(request=AsyncMock(return_value=[{"status": "IN_TRANSIT"}]))
        with patch("main.ActionApiClient", return_value=client):
            await main._handle_message("배송지 변경해주세요 주문번호 100100003", "L-03", "a2")
            text = await main._handle_message("네", "L-03", "a2")
        self.assertIn("변경해 드릴 수 없어요", text)
        self.assertEqual(client.request.await_count, 1)

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

    async def test_return_and_unknown_order_number_get_distinct_guidance(self) -> None:
        main._CONVERSATIONS.clear()
        ret = await main._handle_message("반품해죠", None, "u1")
        unknown = await main._handle_message("주문번호는 모르는데", None, "u1")
        self.assertIn("반품 가능 여부", ret)
        self.assertIn("모르셔도 괜찮아요", unknown)
        self.assertNotEqual(ret, unknown)
        self.assertEqual(main._intent("안 먹고 싶어 반품해줘"), "return")

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
