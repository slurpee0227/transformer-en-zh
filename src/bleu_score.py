from sacrebleu import corpus_bleu


def calculate_bleu(references, hypotheses):
    bleu = corpus_bleu(hypotheses, [references])

    print(f'BLEU Score: {bleu.score:.2f}')

    return bleu.score
