from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_core import answer_question, health_check

app = FastAPI(title="RepoGPT Web UI")

templates = Jinja2Templates(directory="templates")

# serve /static/*
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def api_health():
    return JSONResponse(health_check())


@app.post("/api/ask")
async def api_ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    result = answer_question(question)
    return JSONResponse(result)