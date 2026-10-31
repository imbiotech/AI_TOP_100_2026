from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import dotenv
from fastapi import FastAPI, Request

from action_client import ActionApiClient, ActionApiError
from ai_interpreter import interpret_message


dotenv.load_dotenv()

app = FastAPI()
_ORDER_NUMBER_PATTERN = re.compile(
    r"(?:주문\s*(?:번호|no)|order(?:\s*(?:no|number))?)\s*[:#]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9-]{4,31})",
    re.IGNORECASE,
)
_TEST_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_TEXT_TEST_CASE_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._-])([A-Z]{1,3}-\d{2,3}(?:-[a-z])?)(?![A-Za-z0-9._-])"
)
_TEST_CASE_IDS: OrderedDict[str, tuple[str, float]] = OrderedDict()
_CONVERSATION_TTL_SECONDS = 600
_MAX_CONVERSATIONS = 256


@dataclass
class ConversationState:
    order_number: str | None
    intent: str
    updated_at: float


_CONVERSATIONS: OrderedDict[str, ConversationState] = OrderedDict()


def _simple_text(text: str) -> dict[str, Any]:
    """Return one Kakao SimpleText output within the platform limit."""
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": text[:500]}}],
        },
    }


def _test_case_id(body: dict[str, Any], headers: Any) -> str | None:
    """Read an explicit test-case ID from a supported request location."""
    header_value = headers.get("X-Aitop100-Test-Case-Id")
    if isinstance(header_value, str) and _TEST_CASE_ID_PATTERN.fullmatch(
        header_value
    ):
        return header_value

    user_request = body.get("userRequest")
    candidates: list[Any] = [body.get("test_case_id"), body.get("testCaseId")]
    if isinstance(user_request, dict):
        params = user_request.get("params")
        if isinstance(params, dict):
            candidates.extend(
                params.get(key)
                for key in ("test_case_id", "testCaseId", "tc_id")
            )

    for candidate in candidates:
        if isinstance(candidate, str) and _TEST_CASE_ID_PATTERN.fullmatch(
            candidate
        ):
            return candidate
    return None


def _user_id(body: dict[str, Any]) -> str | None:
    user_request = body.get("userRequest")
    user = user_request.get("user") if isinstance(user_request, dict) else None
    if not isinstance(user, dict):
        return None
    user_id = user.get("id")
    return user_id if isinstance(user_id, str) and user_id else None


def _extract_order_number(text: str) -> str | None:
    match = _ORDER_NUMBER_PATTERN.search(text)
    return match.group(1) if match else None


def _remember_conversation(
    user_id: str | None,
    intent: str,
    order_number: str | None,
) -> None:
    if user_id is None:
        return
    _CONVERSATIONS[user_id] = ConversationState(
        order_number=order_number,
        intent=intent,
        updated_at=time.monotonic(),
    )
    _CONVERSATIONS.move_to_end(user_id)
    while len(_CONVERSATIONS) > _MAX_CONVERSATIONS:
        _CONVERSATIONS.popitem(last=False)


def _known_conversation(user_id: str | None) -> ConversationState | None:
    if user_id is None:
        return None
    state = _CONVERSATIONS.get(user_id)
    if state is None:
        return None
    if time.monotonic() - state.updated_at > _CONVERSATION_TTL_SECONDS:
        del _CONVERSATIONS[user_id]
        return None
    _CONVERSATIONS.move_to_end(user_id)
    return state


def _clear_conversation(user_id: str | None) -> None:
    if user_id is not None:
        _CONVERSATIONS.pop(user_id, None)


def _says_order_number_unknown(text: str) -> bool:
    return "주문번호" in text.replace(" ", "") and any(
        term in text for term in ("모르", "몰라", "기억", "없어")
    )


def _intent(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("병원비", "진료비", "치료비", "배상")):
        return "claim"
    if any(
        term in lowered
        for term in ("배송", "택배", "송장", "미수령", "못 받")
    ):
        return "shipping"
    if any(
        term in lowered
        for term in ("반품", "교환", "변심", "안 먹", "먹기 싫", "먹고 싶지")
    ):
        return "return"
    if any(term in lowered for term in ("주문", "환불", "취소", "결제")):
        return "order"
    if any(term in lowered for term in ("배탈", "복통", "구토", "설사", "아프")):
        return "health"
    return "general"


def _order_status_text(order: dict[str, Any]) -> str:
    status_text = {
        "PAID": "결제 확인",
        "CONFIRMED": "주문 확인",
        "CANCEL_REQUESTED": "취소 요청 접수",
        "CANCELED": "취소",
        "FAILED": "주문 확인 필요 (결제 상태 별도 확인)",
    }.get(order.get("status"), "확인이 필요한 상태")
    return f"주문 상태는 '{status_text}'입니다."


def _shipment_status_text(shipments: list[dict[str, Any]]) -> str:
    if not shipments:
        return "주문은 확인했지만 연결된 배송 정보는 조회되지 않았어요."
    labels = {
        "READY": "출고 준비",
        "IN_TRANSIT": "배송 중",
        "DELIVERED": "배송완료 기록 있음",
        "RETURNED": "반송",
    }
    summary = " / ".join(
        labels.get(item.get("status"), "상태 확인 필요")
        for item in shipments[:3]
        if isinstance(item, dict)
    )
    return (
        f"배송 조회 결과는 {summary}입니다. 배송완료 기록만으로 "
        "실제 수령을 단정할 수는 없어요."
    )


async def _handle_message(
    text: str,
    test_case_id: str | None,
    user_id: str | None,
) -> str:
    previous = _known_conversation(user_id)
    explicit_order_number = _extract_order_number(text)
    rule_intent = _intent(text)
    interpretation = await interpret_message(
        text, previous.intent if previous else None
    )
    intent = rule_intent
    order_number_unknown = _says_order_number_unknown(text)
    if interpretation is not None:
        order_number_unknown = order_number_unknown or (
            previous is not None and interpretation.order_number_unknown
        )
        explicit_order_number = (
            explicit_order_number or interpretation.order_number
        )
        if interpretation.intent != "general":
            intent = interpretation.intent
        # Escalation requires an explicit compensation request by the rules.
        if intent == "claim" and rule_intent != "claim":
            intent = "health"
    if (
        intent == "order"
        and explicit_order_number
        and previous
        and previous.intent in {"health", "claim", "shipping", "order", "return"}
    ):
        intent = previous.intent

    related_previous_order = bool(
        previous
        and (
            intent == previous.intent
            or intent == "claim" and previous.intent in {"health", "claim"}
        )
    )
    order_number = explicit_order_number or (
        previous.order_number if related_previous_order and previous else None
    )
    _remember_conversation(user_id, intent, order_number)

    if intent == "general":
        return (
            "문의 내용을 확인했어요. 상품, 주문·결제, 배송, 환불 중 어떤 도움이 필요한지 "
            "조금 더 구체적으로 알려주세요."
        )

    claim_requested = intent == "claim"
    if intent == "health" and not order_number:
        return (
            "불편을 드려 죄송합니다. 우선 해당 상품은 드시지 말고 보관해 주세요. "
            "현재 몸 상태가 걱정되시면 진료를 받아 주세요. 확인을 위해 주문번호를 알려주시겠어요?"
        )

    if not order_number and order_number_unknown:
        return (
            "주문번호를 모르셔도 괜찮아요. 주문 완료 문자나 알림톡, 주문 내역 화면에서 "
            "확인할 수 있어요. 확인이 어렵다면 상품명과 주문 시기를 알려주세요. "
            "다만 주문번호 없이 주문 정보를 추측해 안내하지는 않습니다."
        )

    if not order_number:
        if intent == "return":
            return (
                "반품 가능 여부는 상품 종류와 수령 상태에 따라 달라 주문 확인이 필요해요. "
                "주문번호를 알려주시면 확인해 드릴게요. 환불이나 반품을 "
                "확정해서 안내하지는 못하는 점 양해 부탁드려요."
            )
        if claim_requested:
            return (
                "상황을 확인해 담당자 검토가 필요할 수 있어요. 먼저 대상 주문번호와 "
                "신고하신 내용, 현재 필요한 도움을 알려주시겠어요?"
            )
        return (
            "확인할 주문번호를 알려주세요. 개인정보 보호를 위해 주문번호를 "
            "추측해 조회하지 않겠습니다."
        )

    if test_case_id is None:
        return (
            "현재 이 채널에서는 주문 상태를 확인할 수 없어요. "
            "주문 조회가 가능한 상담 창구로 문의해 주세요."
        )

    try:
        client = ActionApiClient()
    except ValueError:
        return "액션 API 인증 설정이 없어 주문 조회를 진행하지 못했습니다."

    encoded_order_number = quote(order_number, safe="")
    try:
        order = await client.request(
            "GET",
            f"/api/orders/{encoded_order_number}",
            test_case_id,
        )
        if not isinstance(order, dict):
            return "주문 조회 결과 형식이 올바르지 않아 상태를 안내할 수 없습니다."

        if intent == "shipping":
            shipments = await client.request(
                "GET",
                f"/api/shipments?order_no={encoded_order_number}",
                test_case_id,
            )
            if not isinstance(shipments, list):
                return "배송 조회 결과 형식이 올바르지 않아 상태를 안내할 수 없습니다."
            return _shipment_status_text(shipments)

        if intent in {"health", "claim"}:
            shipments = await client.request(
                "GET",
                f"/api/shipments?order_no={encoded_order_number}",
                test_case_id,
            )
            if not isinstance(shipments, list):
                return "배송 정보 확인이 되지 않아 담당자 검토 요청은 접수하지 않았습니다."
            if not claim_requested:
                return (
                    f"{_order_status_text(order)} "
                    f"{_shipment_status_text(shipments)} "
                    "상품 하자와 증상의 관련성은 확인되지 않았어요. "
                    "남은 상품 사진과 현재 증상을 알려주세요."
                )
            receipt = await client.request(
                "POST",
                "/api/escalations",
                test_case_id,
                {
                    "lane": "LANE3_CLAIM",
                    "reason": (
                        f"주문 {order_number}: 고객이 상품 섭취 후 증상 및 "
                        "병원비/배상 검토를 요청함. 상품 하자와 증상의 인과관계는 "
                        "확인되지 않아 안전 확인 및 정식 검토 요청."
                    ),
                },
            )
            if (
                not isinstance(receipt, dict)
                or not isinstance(receipt.get("escalation_id"), str)
                or receipt.get("request_status") != "SUBMITTED"
            ):
                return (
                    "담당자 검토 요청 결과를 확인할 수 없어 "
                    "접수 완료로 안내하지 않았습니다."
                )
            _clear_conversation(user_id)
            return (
                f"{_order_status_text(order)} 담당자 검토 요청이 접수됐습니다"
                f"(접수번호 {receipt['escalation_id']}). 하자와 증상의 인과관계 "
                "또는 배상 여부가 확정된 것은 아닙니다. 진단서와 진료비 영수증을 "
                "준비해 주세요. 회신 기한은 영업일 3일입니다."
            )

        if intent == "return":
            return (
                f"{_order_status_text(order)} 반품 가능 여부는 상품과 수령 상태에 "
                "따라 달라 아직 확정할 수 없어요. 상품 상태와 사유를 알려주세요."
            )
        return _order_status_text(order)
    except ActionApiError as error:
        if error.status_code == 429:
            if error.retry_after and error.retry_after.isdigit():
                return (
                    f"조회 요청 한도에 도달했어요. {error.retry_after}초 후 다시 "
                    "문의해 주세요. 이번에는 추가 요청을 보내지 않았습니다."
                )
            return "조회 요청 한도에 도달했어요. 잠시 후 다시 문의해 주세요."
        if error.status_code in {500, 503}:
            return "조회 서비스를 일시적으로 사용할 수 없어요. 잠시 후 다시 문의해 주세요."
        if error.status_code == 404:
            return (
                "해당 주문 정보가 조회되지 않았어요. 주문번호를 다시 확인해 주세요. "
                "조회되지 않았다는 이유만으로 결제나 주문 유실을 확정할 수는 없습니다."
            )
        return (
            "주문 조회를 완료하지 못했어요. 결과를 확인하지 못해 처리가 "
            "완료됐다고 안내할 수 없습니다."
        )


@app.post("/")
async def kakao_webhook(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return _simple_text("요청을 읽지 못했습니다. 문의 내용을 다시 보내주세요.")

    user_request = body.get("userRequest") if isinstance(body, dict) else None
    utterance = (
        user_request.get("utterance")
        if isinstance(user_request, dict)
        else None
    )
    if not isinstance(utterance, str) or not utterance.strip():
        return _simple_text("문의 내용을 확인하지 못했습니다. 메시지를 다시 보내주세요.")

    user_id = _user_id(body)
    test_case_id = _test_case_id(body, request.headers)
    match = _TEXT_TEST_CASE_ID_PATTERN.search(utterance)
    if test_case_id is None and match:
        test_case_id = match.group(1)
    if user_id is not None:
        if test_case_id is not None:
            _TEST_CASE_IDS[user_id] = (test_case_id, time.monotonic())
            _TEST_CASE_IDS.move_to_end(user_id)
            while len(_TEST_CASE_IDS) > _MAX_CONVERSATIONS:
                _TEST_CASE_IDS.popitem(last=False)
        else:
            saved = _TEST_CASE_IDS.get(user_id)
            if saved and time.monotonic() - saved[1] <= _CONVERSATION_TTL_SECONDS:
                test_case_id = saved[0]

    response_text = await _handle_message(
        utterance.strip(),
        test_case_id,
        user_id,
    )
    return _simple_text(response_text)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "AI Agent Bot is running!"}
