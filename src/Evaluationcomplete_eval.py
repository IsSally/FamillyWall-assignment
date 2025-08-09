def evaluate_business(business_description, llm_suggestor, tokenizer, nb_suggestions, zeroshot=False):
    #if not is_safe(business_description):
    if not is_safe_with_groq(business_description):
        blocked = {"suggestions": [],
            "status": "blocked",
            "message": "Request contains inappropriate content"}
        print(json.dumps(blocked, indent=2))
        return blocked

    # Generate domains
    if not zeroshot:
      domains = suggest_domains(llm_suggestor, tokenizer, business_description, nb_suggestions)
    else:
      domains = zero_shot_suggest(llm_suggestor, tokenizer, business_description, nb_suggestions)

    # Make sure domains are generated
    if not domains:
        return {
            "suggestions": [],
            "status": "error",
            "message": "Domain generation failed"
        }

    suggestions = [score_domain2(d,business_description) for d in domains]
    result = {
        "suggestions": suggestions,
        "status": "success"
    }
    print(json.dumps(result, indent=2))
    return result
