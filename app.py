from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

class GenerateRequest(BaseModel):
    text_input: str
    max_tokens: int
    bad_words: str = ""
    stop_words: str = ""

@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    url = "http://localhost:8000/v2/models/ensemble/generate"
    payload = request.dict()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Request failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return response.json()