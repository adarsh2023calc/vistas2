from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output": ""})

@app.post("/", response_class=HTMLResponse)
async def debug(request: Request, code: str = Form(...)):
    # Simulate AI debug logic
    output = f"Simulated output:\n\nNo bugs found in:\n\n{code}"
    return templates.TemplateResponse("index.html", {"request": request, "output": output, "code": code})
