from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You are an expert in Vietnamese public procurement market intelligence. Vietnamese public procurement operates on the basis of competitive bidding, where multiple contractors (sellers) bid their products, and investors (buyers) will choose products based on technical requirements and pricing. Most influential factors include: product manufacturer, country of origin, and region of origin (proxy for quality), along with bid price (lowest wins). We are developing an application to support contractors (users) by recommending optimal pricing strategies that maximize profit and chances of winning. We will give you a summary of historical data, information on user's product, and our model's predictions of the optimal pricing strategy. Your responsibility is to (1) closely analyze all given information and (2) write a concise answer (under 200 words) that recommends the best unit price or range of unit price and give your reasoning. Structure your answer in clear sections and bullet points. Answer in English, and note that your answer is user-facing. IMPORTANT: BASED ALL YOUR JUDGEMENTS ON THE GIVEN INFORMATION. In your answer, include the unit (VND) for all mentions of prices, with no decimal, with ',' to separate every 3 digits."

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
	for event in stream:
		if event.type == "response.output_text.delta":
			yield event.delta



