from fastapi import FastAPI, Request, Form,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.utilities.serpapi import SerpAPIWrapper
from pydantic import BaseModel
from passlib.context import CryptContext
from db import users_collection  # Use relative import if db.py is in the same directory
import requests
import os
from dotenv import load_dotenv





# Initialize the app and load environment variables
app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()  # Make sure to load the .env file for GROQ_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

serp_tool = SerpAPIWrapper()





def search_github_issues(error: str) -> str:
    url = f"https://api.github.com/search/issues?q={error}+in:title,body+type:issue"
    headers = {
        "Accept": "application/vnd.github+json",
        # Optional: Add token for higher rate limit
        # "Authorization": "Bearer YOUR_GITHUB_TOKEN"
    }
    response = requests.get(url, headers=headers)
    items = response.json().get("items", [])[:5]

    if not items:
        return "No related GitHub issues found."

    results = []
    for item in items:
        results.append(f"- [{item['title']}]({item['html_url']})")

    return "\n".join(results)



search_tool = Tool(
    name="Web Search",
    func=serp_tool.run,
    description="Google search for coding errors, solutions, docs"
)

github_tool = Tool(
    name="GitHub Search",
    func=search_github_issues,
    description="Search relevant GitHub issues"
)



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
def ask_gpt(code, model,error):
    # Setup the model
    llm = ChatGroq(model=model)

    # Prompt
    system_prompt = (
        "You are a helpful AI that reviews code, finds bugs or issues, "
        "and provides corrected code with explanations.\n\n"
        "We will try to debug using the following steps:\n"
        "1. Identify any errors in the code\n"
        "2. Understand the user's intent\n"
        "3. Look for syntax errors\n"
        "4. Check for semantic correctness\n"
        "5. Evaluate logical correctness\n"
        "6. Suggest improvements if needed\n"
        "7. Find any security problems like SQL Injection\n"
        "8. Output the corrected code with explanation\n\n"
        "=== User Code ===\n"
        f"{code}\n\n"
        "=== Error Message ===\n"
        f"{error}\n\n"
        "=== Start your analysis below ===\n"
    )

    tools = [
    Tool(name="GitHub Search", func=search_github_issues, description="Search relevant GitHub issues"),
    Tool(name="Web Search", func=serp_tool.run, description="Google search for coding errors, solutions, docs"),
     ]

    # Step 1: Create the ReAct agent with tools
    agent = initialize_agent(
    tools=tools,
    llm=ChatGroq(model="mixtral-8x7b-32768"),  # or "llama3-70b-8192"
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)
    response = agent.run(prompt)

    return response
    

# Homepage route (GET)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output": "", "model": "","error":""})

# Form submission (POST)
@app.post("/", response_class=HTMLResponse)
async def ask_groq(request: Request, code: str = Form(...), model: str = Form(...),language:str=Form(...),error:str=Form(...)):
    print(f"Selected Model: {model}")  # Debugging line
    output = ask_gpt(code, model,error)  
    code_debugging_output= execute_code(language,code)
    print(code_debugging_output)
    return templates.TemplateResponse("index.html", {"request": request, "output": output, "code": code, "model": model,"debug":code_debugging_output})


# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User model for signup/login
class User(BaseModel):
    username: str
    password: str

# Register route
@app.post("/register")
def register(user: User):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = pwd_context.hash(user.password)
    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password
    })
    return {"message": "User registered successfully"}

# Login route
@app.post("/login")
def login(user: User):
    existing_user = users_collection.find_one({"username": user.username})
    if not existing_user or not pwd_context.verify(user.password, existing_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {"message": "Login successful"}
