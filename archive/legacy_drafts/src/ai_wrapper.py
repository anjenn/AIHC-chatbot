from openai import OpenAI

client = OpenAI()

def run_openai(prompt, model=OPENAI_MODEL, temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()