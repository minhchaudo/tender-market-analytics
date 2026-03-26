from openai import OpenAI
import os

SYSTEM_PROMPT = (
	"You are a expert tender market analytics about Vietnamese public investment sector. "
	"Answer clearly and concise, and avoid hallucinations."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
	try:
		for event in stream:
			if event.type == "response.output_text.delta":
				yield event.delta
	except Exception as e:
		yield f"LLM error: {e}"
