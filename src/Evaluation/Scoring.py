#Function to evaluate on the test set using bleu score
def Evaluate_bleu_n_perplexity(model, tokenizer, test_dataset):
  bleu_scores = []
  all_refs = []  # for corpus BLEU
  all_hyps = []
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  # Keep model config in sync
  model.config.eos_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  device = next(model.parameters()).device

  test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
    device=torch.device(device)
  )
  model.eval()
  nll_sum = 0.0
  tok_count = 0
  with torch.inference_mode():
    for ex in test_dataset:
        # turn pre-tokenized lists into a 1×seq_len batch
        ids   = torch.tensor(ex["input_ids"]).unsqueeze(0)
        mask  = torch.tensor(ex["attention_mask"]).unsqueeze(0)
        labels = ex["labels"]
        labels_b = labels.unsqueeze(0)

        # Perplexity
        out = model(input_ids=ids,
                        attention_mask=mask,
                        labels=labels_b)  # HF computes CE mean over non-ignored tokens
        fill_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # count target tokens (ignore -100 if present)
        if (labels == -100).any():
            num_toks = (labels != -100).sum().item()
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = fill_id
        else:
            pad = fill_id
            num_toks = (labels != pad).sum().item() if pad is not None else labels.numel()
            labels_for_decode = labels
        nll_sum += out.loss.item() * max(1, num_toks)
        tok_count += max(1, num_toks)
        #Bleu
        # generate top-1
        out_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            #max_length=128,
            max_new_tokens=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            #early_stopping=True,
        )[0]

        
        if (labels == -100).any():
            labels = labels.clone()
            labels[labels == -100] = tokenizer.eos_token_id

        # decode
        ref_str = tokenizer.decode(labels, skip_special_tokens=True) #decode the reference
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)

        # compute BLEU against the single reference
        ref = ref_str.split()
        hyp = pred.split()
        bleu_scores.append(sentence_bleu([ref], hyp))
        # collect for corpus BLEU
        all_refs.append([ref])   # note: list-of-list for possible multiple refs
        all_hyps.append(hyp)

  # Report
  avg_bleu = sum(bleu_scores) / len(bleu_scores)
  corpus_bleu_score = corpus_bleu(all_refs, all_hyps)
  ppl = math.exp(nll_sum / max(1, tok_count))
  #print(f"Average BLEU over {len(bleu_scores)} examples: {avg_bleu:.4f}")
  print(f"Corpus BLEU: {corpus_bleu_score:.4f}")
  print(f"Perplexity: {ppl:.4f}")
  return corpus_bleu_score, ppl

def calculate_avg_std_scores(model, tokenizer, nb_repetition, dataset):
  bleu_score = []
  ppl_score = []
  llm_score = []
  for i in range(nb_repetition):
    bleu, ppl =  Evaluate_bleu_n_perplexity(model, tokenizer, dataset)
    llm = Evaluate_llm_score_on_dataset(model, tokenizer, dataset)
    bleu_score.append(bleu)
    ppl_score.append(ppl)
    llm_score.append(llm)
  avg_bleu = stats.mean(bleu_score)
  avg_ppl = stats.mean(ppl_score)
  avg_llm = stats.mean(llm_score)
  std_bleu = stats.stdev(bleu_score)
  std_ppl = stats.stdev(ppl_score)
  std_llm = stats.stdev(llm_score)
  print(f"The average: bleu score {avg_bleu}, perplexity score {avg_ppl}, and llm score {avg_llm}.")
  print(f"The std: bleu score {std_bleu}, perplexity score {std_ppl}, and llm score {std_llm}.")
  return avg_bleu, avg_ppl, avg_llm, std_bleu, std_ppl, std_llm
