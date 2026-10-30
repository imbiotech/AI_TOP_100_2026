from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/")
async def kakao_webhook(request: Request):
    body = await request.json()
    # 카카오톡 사용자가 보낸 메시지 추출
    user_message = body.get("userRequest", {}).get("utterance", "")
    
    # 테스트용 응답 메시지
    response_text = f"에이전트가 받은 메시지: {user_message}"
    
    # 카카오톡 오픈빌더 규격 JSON 반환
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": response_text
                    }
                }
            ]
        }
    }

@app.get("/")
async def root():
    return {"message": "AI Agent Bot is running!"}

