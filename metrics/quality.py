from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def repetition_rate(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    unique = set(tokens)
    return round(1 - len(unique) / len(tokens), 4)

def bleu_score(hypothesis: str, reference: str) -> float:
    ref_tokens = [reference.split()]
    hyp_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method1
    return round(sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie), 4)