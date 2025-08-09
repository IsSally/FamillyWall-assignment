#GROQ_API_KEY = "your-groq-api-key"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def score_domain2(domain, business_description):
    prompt = f"""
You are a brand expert evaluating domain names for businesses.

Here is the business description:
"{business_description}"

Evaluate the following domain name: "{domain}"

Score it based on:
1. Memorability
2. Pronounceability
3. Brevity
4. Brandability
5. Relevance to the business description
6. Avoids ambiguity

Each is rated from 1 (Poor) to 5 (Excellent).

Return only this JSON format:
{{
  "domain": "{domain}",
  "score": <total_score>,
  "confidence": <score/30, rounded to 2 decimals>
}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a domain evaluation expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Extract the first valid JSON block
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            json_block = match.group()
            result = json.loads(json_block)
            return {
                "domain": result["domain"],
                "confidence": round(result["score"] / 30, 2)
            }
        else:
            raise ValueError("No JSON block found")

    except Exception as e:
        print("Error scoring domain:", e)
        print("Response:", response.text)
        return {"domain": domain, "confidence": 0.0}
