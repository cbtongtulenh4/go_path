import os
import asyncio
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    async def get_structured_response(self, system_prompt: str, user_prompt: str):
        """
        Calls Groq API and expects a JSON response.
        """
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # In a production system, we would log this properly
            print(f"Error in LLMService: {e}")
            return None
