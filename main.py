from fastapi import FastAPI, Request, Form,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import requests
import os
from dotenv import load_dotenv
from groq import Groq

# Initialize the app and load environment variables
app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()  # Make sure to load the .env file for GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




def execute_code(language,code):
    try:

        response = requests.post(
            "https://emkc.org/api/v2/piston/execute",
            json={
                "language": language,
                "source": code
            }
        )
        
        return response.json()
    except Exception as e:
        return HTTPException(status_code=500,detail=str(e))


# Helper function to interact with the GPT API
def ask_gpt(code, model):

    prompt = (
        "You are a helpful AI that reviews code, finds bugs or issues, "
        "and provides corrected code with explanations.\n"
        "We will try to debug using the following steps:\n"
        "1. Identify the programming language.\n"
        "2. Understand the user's intent.\n"
        "3. Look for syntax errors.\n"
        "4. Check for semantic correctness.\n"
        "5. Evaluate logical correctness.\n"
        "6. Suggest improvements if needed.\n"
        "7. Find any security problems like SQL Injection\n"
        "7. Output the corrected code with explanation.\n"
    )

   
    messages= [{"role": "system", "content": prompt},{"role": "user", "content": code}]
      

    

    client = Groq()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    output=" "
    for chunk in completion:
        if chunk.choices[0].delta.content:
            output+=chunk.choices[0].delta.content
        else:
            output+=" "
    
    return output
    

# Homepage route (GET)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output": "", "model": ""})

# Form submission (POST)
@app.post("/", response_class=HTMLResponse)
async def ask_groq(request: Request, code: str = Form(...), model: str = Form(...),language:str=Form(...)):
    print(f"Selected Model: {model}")  # Debugging line
    output = ask_gpt(code, model)  # Get the GPT response
    code_debugging_output= execute_code(language,code)
    print(code_debugging_output)
    return templates.TemplateResponse("index.html", {"request": request, "output": output, "code": code, "model": model,"debug":code_debugging_output})
