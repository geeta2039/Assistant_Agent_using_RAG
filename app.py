from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from main import retrieve_relevant_chunks, answer_question

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": "", "query": "", "model": "flan-base"})

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, query: str = Form(...), model: str = Form("flan-base")):
    try:
        relevant = retrieve_relevant_chunks(query)
        answer = answer_question(query, relevant, model=model)
    except Exception as e:
        answer = f"Error: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "query": query, "model": model})
