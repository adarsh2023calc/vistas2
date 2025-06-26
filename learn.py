

from langchain_groq import ChatGroq

import json





def AI_Code_Reviewer(code):
    prompt = f'''You are a strict code reviewer. 
                    user_prompt = f"""
        Analyze the following code and return a score from 0 to 100 based only on:
        1. Code readability
        2. Naming conventions
        3. Modularity and reuse
        4. Commenting and documentation
        5. Error handling (if applicable)

        Use a reasoning-based approach (Chain of Thought) to assess each category out of 20,
        then sum them for a final score. Output ONLY this exact JSON format:

        {{
        "readability": "...",
        "efficiency": "...",
        "modularity": "...",
        "comments": "...",
        "overall_score": <number>
        }}

        Do not include markdown, explanations, or any extra output.
        Code:
        ```python
        {code}
        ```
    '''

    try:
        llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
        response = llm.invoke(prompt)
        print(response)
        # Parse JSON response from LLM
        result = json.loads(response.content)
        
        
        return result
    
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

        
    
    

