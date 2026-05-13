import torch


def beam_search_decode(model, src_tensor, bos_idx, eos_idx, device, beam_width=3, max_length=50):
    sequences = [([bos_idx], 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            tgt_tensor = torch.tensor([seq]).to(device)

            with torch.no_grad():
                output = model(src_tensor, tgt_tensor)

            probs = torch.softmax(output[:, -1], dim=-1)

            top_probs, top_indices = probs.topk(beam_width)

            for i in range(beam_width):
                token = top_indices[0][i].item()
                candidate = seq + [token]
                candidate_score = score - torch.log(top_probs[0][i]).item()

                all_candidates.append((candidate, candidate_score))

        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

        if sequences[0][0][-1] == eos_idx:
            break

    return sequences[0][0]
