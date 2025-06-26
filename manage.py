from typing import Dict, List, Optional
from datetime import datetime, timedelta
from db import feedback_collection
from textblob import TextBlob

class PromptManager:
    def __init__(self):
        self.base_prompt_template = (
            "You are an expert coding assistant who teaches kids. Your task is to review code, identify bugs or issues, and provide the corrected code along with explanations.\n\n"
        "Follow these exact steps when debugging:\n"
        "1. Identify any errors in the code.\n"
        "2. Understand the user's intended functionality.\n"
        "3. Detect syntax errors.\n"
        "4. Check for semantic correctness.\n"
        "5. Verify logical correctness.\n"
        "6. Suggest improvements where necessary.\n"
        "7. Identify security vulnerabilities (e.g., SQL Injection).\n\n"
        "Rules:\n"
        "- If you are CONFIDENT and can directly correct the code without external help, SKIP Thought/Action steps and IMMEDIATELY output the Corrected Code.\n"
        "- If you NEED to search for solutions, first write:\n"
        "  Thought: [Explain why you need to search.]\n"
        "  Action: [Choose ONLY one: GitHub Search or Web Search]\n"
        "  Action Input: [What to search for]\n\n"
        "- NEVER mix Thought and Corrected Code together.\n\n"
        "When providing the final fix:\n"
        "**Corrected Code:**\n"
        "```[language]\n"
        "[your corrected code]\n"
        "```\n\n"
        "**Explanation:**\n"
        "[Explain clearly what was wrong and how you fixed it.]\n\n"
        )
        self.dynamic_sections = {
            'error_patterns': '',
            'solution_patterns': '',
            'best_practices': ''
        }
        self.update_interval = timedelta(hours=1)
        self.last_update = None

    def _analyze_feedback_patterns(self) -> Dict[str, List[str]]:
        """Analyze feedback data to extract successful patterns and common issues."""
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)}
                }
            },
            {
                "$group": {
                    "_id": "$analysis.error_type",
                    "positive_patterns": {
                        "$push": {
                            "$cond": [
                                {"$gt": ["$analysis.sentiment_score", 0.5]},
                                "$dpo_data.preferred_response",
                                "$$REMOVE"
                            ]
                        }
                    },
                    "negative_patterns": {
                        "$push": {
                            "$cond": [
                                {"$lt": ["$analysis.sentiment_score", 0]},
                                "$dpo_data.non_preferred_response",
                                "$$REMOVE"
                            ]
                        }
                    },
                    "avg_sentiment": {"$avg": "$analysis.sentiment_score"}
                }
            }
        ]
        
        results = feedback_collection.aggregate(pipeline)
        patterns = {
            'successful_patterns': [],
            'avoid_patterns': []
        }
        
        for result in results:
            if result['avg_sentiment'] > 0.5 and result['positive_patterns']:
                patterns['successful_patterns'].extend(result['positive_patterns'][:3])
            elif result['avg_sentiment'] < 0 and result['negative_patterns']:
                patterns['avoid_patterns'].extend(result['negative_patterns'][:3])
        
        return patterns

    def _update_dynamic_sections(self):
        """Update dynamic sections of the prompt based on feedback analysis."""
        patterns = self._analyze_feedback_patterns()
        
        # Update error patterns section
        if patterns['avoid_patterns']:
            self.dynamic_sections['error_patterns'] = "\nCommon pitfalls to avoid:\n" + \
                "\n".join(f"- Avoid: {pattern}" for pattern in patterns['avoid_patterns'][:3])
        
        # Update solution patterns section
        if patterns['successful_patterns']:
            self.dynamic_sections['solution_patterns'] = "\nSuccessful solution patterns:\n" + \
                "\n".join(f"- Consider: {pattern}" for pattern in patterns['successful_patterns'][:3])

    def get_prompt(self, code: str, error: str,grade:str) -> str:
        """Get the current prompt template, updating if necessary."""
        current_time = datetime.utcnow()
        
        # Update dynamic sections if needed
        if not self.last_update or (current_time - self.last_update) > self.update_interval:
            self._update_dynamic_sections()
            self.last_update = current_time

        # Combine base template with dynamic sections
        prompt = self.base_prompt_template
        
        for section_content in self.dynamic_sections.values():
            if section_content:
                prompt += section_content + "\n\n"

        # Add the current context
        prompt += (
            f"Imagine you are teaching to {grade} student. explain to them like in a way teachers explain to them"
            f"=== User Code ===\n"
            f"{code}\n\n"
            f"=== Error Message ===\n"
            f"{error}\n\n"
            f"=== Begin your analysis below ===\n"
        )

        return prompt