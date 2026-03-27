from openai import OpenAI
import os
from dotenv import load_dotenv
from get_summary import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)

def llm(prompt):
	stream = client.responses.create(
		model="gpt-5-nano",
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
