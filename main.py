from fastapi import FastAPI, Request, Form,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from groq import Groq
from langchain.tools import Tool
import json
from langchain.utilities.serpapi import SerpAPIWrapper
from pydantic import BaseModel
from passlib.context import CryptContext
from langchain.agents import initialize_agent, Tool, AgentType
from db import users_collection,feedback_collection , store_feedback
import requests
import os
import torch  
from rlhf import analyze_feedback,update_model_weights
from dotenv import load_dotenv
from datetime import datetime, timedelta
from learn import AI_Code_Reviewer
from manage import PromptManager



# Global variables to store current debugging session info

   



current_debug_session = {
    'code': '',
    'error': '',
    'output': ''
    }

# Pydantic model for request body
class LearnRequest(BaseModel):
    fix: str

class Feedback(BaseModel):
    feedback_type: str
    feedback_text: str

# User model for signup/login
class User(BaseModel):
    username: str
    password: str


# Global variables to store current debugging session info


# Initialize the app and load environment variables
app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()  # Make sure to load the .env file for GROQ_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

serp_tool = SerpAPIWrapper()
prompt_manager = PromptManager()


class Feedback(BaseModel):
    feedback_type: str
    feedback_text: str
    preferred_response: str | None = None
    non_preferred_response: str | None = None
    comparison_metrics: dict | None = None  # Additional metrics for comparing responses
    context_info: dict | None = None  # Context in which the responses were generated







import requests

def search_stackoverflow(query: str, max_results: int = 5):
    """
    Search Stack Overflow for programming questions.
    """
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": max_results
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code}"

    data = response.json()
    results = []
    for item in data.get("items", []):
        title = item["title"]
        link = item["link"]
        results.append(f"{title}\n{link}")
    return "\n\n".join(results)


@app.post("/submit_feedback")
async def submit_feedback(feedback: Feedback):
   

    # Analyze feedback
    feedback_analysis = analyze_feedback(
        feedback.feedback_text,
        current_debug_session['code'],
        current_debug_session['error']
    )
    print(feedback_analysis)
    
    # Add DPO-specific analysis for pairwise feedback
    if feedback.preferred_response and feedback.non_preferred_response:
        feedback_analysis['is_pairwise'] = True
        feedback_analysis['comparison_metrics'] = feedback.comparison_metrics or {}
        feedback_analysis['context_info'] = feedback.context_info or {}
    
    # Store feedback with analysis and DPO data
    success = store_feedback(
        feedback_type=feedback.feedback_type,
        feedback_text=feedback.feedback_text,
        code=current_debug_session['code'],
        error=current_debug_session['error'],
        output=current_debug_session['output'],
        preferred_response=feedback.preferred_response,
        non_preferred_response=feedback.non_preferred_response,
        comparison_metrics=feedback.comparison_metrics,
        context_info=feedback.context_info
    )
    
    if success:
        # Update model weights based on feedback
        update_model_weights(feedback_analysis)
        
        return JSONResponse(
            content={
                "message": "Feedback submitted and analyzed successfully",
                "analysis": feedback_analysis
            },
            status_code=200
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to store feedback due to")
        
    

@app.get("/dpo-training-data")
async def get_dpo_training_data():
    try:
        # Query MongoDB for pairwise feedback data
        pipeline = [
            {
                "$match": {
                    "dpo_data.is_pairwise": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "preferred": "$dpo_data.preferred_response",
                    "non_preferred": "$dpo_data.non_preferred_response",
                    "code_context": "$code",
                    "error_context": "$error",
                    "feedback_type": 1,
                    "analysis": 1
                }
            }
        ]
        
        dpo_data = list(feedback_collection.aggregate(pipeline))
        return JSONResponse(content={"dpo_training_pairs": dpo_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/feedback-trends")
async def get_feedback_trends():
    try:
        # Get feedback trends over time (last 30 days)
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": datetime.utcnow() - timedelta(days=30)
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "sentiment_avg": {"$avg": "$analysis.sentiment_score"},
                    "positive_count": {
                        "$sum": {"$cond": [{"$eq": ["$feedback_type", "positive"]}, 1, 0]}
                    },
                    "negative_count": {
                        "$sum": {"$cond": [{"$eq": ["$feedback_type", "negative"]}, 1, 0]}
                    },
                    "total_count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        trends = list(feedback_collection.aggregate(pipeline))
        
        return JSONResponse(content={
            "daily_trends": [
                {
                    "date": day["_id"],
                    "average_sentiment": day["sentiment_avg"],
                    "positive_ratio": day["positive_count"] / day["total_count"],
                    "negative_ratio": day["negative_count"] / day["total_count"],
                    "total_feedback": day["total_count"]
                } for day in trends
            ],
            "trend_analysis": {
                "total_days": len(trends),
                "sentiment_trend": "improving" if len(trends) > 1 and trends[-1]["sentiment_avg"] > trends[0]["sentiment_avg"] else "declining",
                "feedback_volume_trend": sum(day["total_count"] for day in trends) / len(trends) if trends else 0
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    



@app.get("/feedback-analytics")
async def get_feedback_analytics():
    try:
        # Get aggregated statistics from MongoDB
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_count": {"$sum": 1},
                    "avg_sentiment": {"$avg": "$analysis.sentiment_score"},
                    "positive_count": {
                        "$sum": {"$cond": [{"$eq": ["$feedback_type", "positive"]}, 1, 0]}
                    },
                    "negative_count": {
                        "$sum": {"$cond": [{"$eq": ["$feedback_type", "negative"]}, 1, 0]}
                    }
                }
            }
        ]
        overall_stats = list(feedback_collection.aggregate(pipeline))[0]

        # Get top error types
        error_pipeline = [
            {"$group": {"_id": "$analysis.error_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        top_errors = list(feedback_collection.aggregate(error_pipeline))

        # Get code complexity patterns
        complexity_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_code_length": {"$avg": "$analysis.code_length"},
                    "error_handling_count": {
                        "$sum": {"$cond": ["$analysis.complexity_metrics.has_error_handling", 1, 0]}
                    },
                    "commented_code_count": {
                        "$sum": {"$cond": ["$analysis.complexity_metrics.has_comments", 1, 0]}
                    }
                }
            }
        ]
        complexity_stats = list(feedback_collection.aggregate(complexity_pipeline))[0]

        return JSONResponse(content={
            "overall_statistics": {
                "total_feedback": overall_stats["total_count"],
                "average_sentiment": overall_stats["avg_sentiment"],
                "positive_feedback_ratio": overall_stats["positive_count"] / overall_stats["total_count"],
                "negative_feedback_ratio": overall_stats["negative_count"] / overall_stats["total_count"]
            },
            "top_error_types": {
                error["_id"]: error["count"] for error in top_errors
            },
            "code_patterns": {
                "average_code_length": complexity_stats["avg_code_length"],
                "error_handling_percentage": (complexity_stats["error_handling_count"] / overall_stats["total_count"]) * 100,
                "commented_code_percentage": (complexity_stats["commented_code_count"] / overall_stats["total_count"]) * 100
            },
            "improvement_suggestions": [
                f"Focus on improving {error['_id']} handling" for error in top_errors
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
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







def clean_llm_output_with_groq(raw_output):
    prompt = f"""The following text is a raw output from a language model that failed to parse.
        It may contain error messages or parsing failure notes.
        Please remove all mentions of parsing errors or exceptions and return only the meaningful assistant response.

        Raw Output:
        \"\"\"{raw_output}\"\"\"

        Cleaned Response:"""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

    response = client.chat.completions.create(
        model="llama3-70b-8192", 
        messages=[
            {"role": "system", "content": "You are an assistant that cleans malformed LLM output."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


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
def ask_gpt(code, model,error,grade):
     # Store current debugging session info for feedback
    current_debug_session['code'] = code
    current_debug_session['error'] = error
    

    # Setup the model
    llm_model = ChatGroq(model=model)

   
    prompt = prompt_manager.get_prompt(code, error,grade)
    tools = [
    Tool(name="GitHub Search", func=search_github_issues, description="Search relevant GitHub issues"),
    Tool(name="Web Search", func=serp_tool.run, description="Google search for coding errors, solutions, docs"),
    Tool(name="Stack Overflow Search", func=search_stackoverflow, description="Search Stack Overflow for relevant programming questions")
     ]

    # Step 1: Create the ReAct agent with tools
    agent = initialize_agent(
    tools=tools,
    llm=llm_model,  
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)

    try:
        response = agent.run(input=prompt)
        current_debug_session['output'] = response

    except ValueError as e:
        # The ValueError contains the raw LLM output that couldn't be parsed
        print("⚡ Output parsing failed. Capturing raw output.")
        response = str(e)
        response = clean_llm_output_with_groq(response)
        current_debug_session['output']= response
    
    print(current_debug_session)
    return response
    

# Homepage route (GET)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "code":"","output": "", "model": "","error":""})

# Form submission (POST)
@app.post("/", response_class=HTMLResponse)
async def ask_groq(request: Request, code: str = Form(...), model: str = Form(...),language:str=Form(...),error:str=Form(...),grade:str=Form(...)):
    print(f"Selected Model: {model}")  # Debugging line
    output = ask_gpt(code, model,error,grade)  
    if len(code.split()) > 1000:
        code = "\n".join(code.split("\n")[:100]) + "\n# Code truncated due to size limit"
    code_debugging_output= execute_code(language,code)
    code_quality_response = AI_Code_Reviewer(code)
    return templates.TemplateResponse("index.html", {"request": request, "output": output, "code": code, "model": model,"debug":code_debugging_output,"code_quality":code_quality_response})



# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



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


@app.post("/learn")

async def learn(request: LearnRequest):
    fix = request.fix
    
    # Construct a Groq query to search for educational content
    prompt = f"""
        You're an expert programming educator.

        A user encountered a bug and this is the suggested fix:

        \"\"\"{fix}\"\"\"

        Give 5 helpful Educational resources[Links should be working and not misplaced] that explain:
        - Why the error occurred
        - How the fix solves the issue
        - Related concepts a developer should understand
        - Only embed URLs that allow iframing.
        

        Return the result as a JSON **list** of objects, each with:
        - "title": A short, helpful title for the resource
        - "url": A valid link to the resource

        Your output should look like this:
        [
        {{"title": "Understanding Null Pointer Exceptions", "url": "https://example.com/null-pointer"}},
        ...
        ]
        Only output this JSON list. Do not include any extra text.
        """


    # Query Groq for educational resource

    try:
        llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")
        response = llm.invoke(prompt)
        # Parse JSON response from LLM
        result = json.loads(response.content)
        print(result)
        
        return {"resources":result}
    
    except Exception as e:
        print(f"Initial parsing error: {e}")

        try:
            # Retry using the LLM to fix the malformed JSON
            fix_prompt = f"""The following text was supposed to be a JSON response but failed to parse:
            Please correct it and return a valid JSON object only. No explanation needed."""
            
            fixed_response = llm.invoke(fix_prompt)
            result = json.loads(fixed_response.content)
            print("Fixed using LLM:", result)
            return result

        except Exception as inner_e:
            print(f"Fix attempt failed: {inner_e}")
            return {"error": f"Original error: {e}", "fix_attempt_error": str(inner_e)}





