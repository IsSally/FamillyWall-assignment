
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def score_domain2(domain, business_description):
    prompt = f"""
You are a brand expert evaluating domain names for businesses.

Task: score the domain for the business and ALSO report how confident you are in the score you assigned (not a formula).

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
Then:
- total_score = sum of the six criterion scores (integer 6–30).
- score = <score/30, rounded to 2 decimals>
- confidence = your self-rated certainty in the total_score on a 0–1 scale with two decimals, where:
  0.90–1.00 = extremely confident (clear, unambiguous, strong fit)
  0.60–0.89 = moderately confident
  0.30–0.59 = low confidence (ambiguous/weak fit)
  0.00–0.29 = very low (insufficient info or highly ambiguous)

Return only this JSON format:
{{
  "domain": "{domain}",
  "score": <score>,
  "confidence": <float 0-1 with two decimals>
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
                "score": result["score"],
                "confidence": result["confidence"]
            }
        else:
            raise ValueError("No JSON block found")

    except Exception as e:
        print("Error scoring domain:", e)
        print("Response:", response.text)
        return {"domain": domain, "score": 0.0, "confidence": 0.0}


def Evaluate_llm_score_on_dataset(model, tokenizer, test_dataset):
  scores = []

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  device = next(model.parameters()).device

  test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
    device=torch.device(device)
  )
  model.eval()
  with torch.inference_mode():
    for ex in test_dataset:
        # turn pre-tokenized lists into a 1×seq_len batch
        ids   = torch.tensor(ex["input_ids"]).unsqueeze(0)
        mask  = torch.tensor(ex["attention_mask"]).unsqueeze(0)
        labels = ex["labels"]
        
        out_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )[0]


        if (labels == -100).any():
            labels = labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id

        # decode
        ref_ip = tokenizer.decode(ex["input_ids"], skip_special_tokens=True) 
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        ref = ref_ip.split()
        hyp = pred.split()
        scores.append(score_domain2(hyp, ref)['score'])

  # Report
  avg_score = stats.mean(scores)
  print(f"The average LLM score: {avg_score}.")
  return avg_score

