from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a pricing assistant.

Rewrite the explanation into a SHORT, business-friendly format.

STRICT RULES:
- Do NOT change any numbers
- Do NOT add new information
- Do NOT use technical terms (e.g. percentile, intersection)
- Focus on meaning, not process

FORMAT:

Recommendation  
- <optimal price> (range: <range>)  

Why  
- Maximum 3 bullet points  
- Each bullet must add a new insight  
- Focus on market behavior and decision logic  

Reliability  
- 1 short sentence only  

Keep it concise and easy to scan."""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)

def llm(prompt):
	stream = client.responses.create(
		model="gpt-5-mini",
		input=[
			{
				"role": "system",
				"content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
			},
			{
				"role": "user",
				"content": [{"type": "input_text", "text": prompt}],
			},
		],
		stream=True
	)
	for event in stream:
		if event.type == "response.output_text.delta":
			yield event.delta
