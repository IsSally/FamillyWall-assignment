from transformers import AutoModelForCausalLM

# 3. Define a generic suggestion helper to generate domains using the trained models
def suggest_domains(model, tokenizer, biz_desc, n=3):
    prompt = f"Business: {biz_desc}\nDomain suggestions:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        num_return_sequences=n,
        temperature=0.7,
        do_sample=True,
    )
    return [
        tokenizer.decode(o, skip_special_tokens=True)
                 .split("Domain suggestions:")[-1]
                 .strip()
        for o in outputs
    ]

# Zero shot function
def zero_shot_suggest(model, tokenizer, biz_desc, num_return_sequences=3):
    #prompt = f"Business: {biz_desc}\nDomain suggestions:"
    prompt = (
    f"Business: {biz_desc}\n"
    f"Give me exactly {num_return_sequences} domain names only, separated by commas, no other text.\n"
    "Domains:"
)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        do_sample=True,
    )

    suggestions = []
    for o in outputs:
        text = tokenizer.decode(o, skip_special_tokens=True)
        # grab only what comes after our “Domains:” marker
        after = text.split("Domains:")[-1]
        # split by commas or newlines, strip whitespace/punctuation
        candidates = re.split(r"[,\n]+", after)
        for cand in candidates:
            print("cand: ",cand)
            c = cand.strip().strip(".-–* ")  # clean leading bullets/punctuation
            print("c: ",c)
            # keep only things that look like a domain
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9\-_]+\.[A-Za-z]{2,}$", c):
                suggestions.append(c)
        # once we have num_return_sequences suggestions, stop
        if len(suggestions) >= num_return_sequences:
            break
    return suggestions[:num_return_sequences]