import requests

GROQ_API_KEY = getpass("Enter your Groq key:")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def is_safe_with_groq(text):
    prompt = f"""
You are a content safety evaluator.

Your task is to decide if the following text is unsafe due to containing:
- sexually explicit content
- hate speech or racism
- graphic violence
- illegal activity (e.g., drugs, weapons, gambling)
- harassment or abuse

Respond with one word only: "safe" or "unsafe".

Text: "{text}"
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a content safety classifier."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip().lower()
        return result == "safe"
    except Exception as e:
        print("Moderation error:", e)
        print("Response:", response.text)
        return False
