import re
import numpy as np

class SentenceEmbedder:
    vocab_embeddings = {
        "i":          np.array([-1.0,  1.0]),
        "lot":        np.array([ 0.0,  2.0]),
        "love":       np.array([ 1.0,  3.0]),
        "chocolate":  np.array([ 2.0,  4.0]),
        "milk":       np.array([ 3.0,  5.0])
    }

    filler_words = {"i", "and", "as", "for", "it", "or", "maybe"}

    attention_weights = {
        "i":          0.5,
        "lot":        1.0,
        "love":       0.7,
        "chocolate":  0.9,
        "milk":       0.3
    }

    def __init__(self):
        pass

    def _normalize(self, sentence):
        cleaned = re.sub(r"[^\w\s]", "", sentence.lower())
        words = cleaned.split()

        return words

    def _get_embedding(self, word):
        return self.vocab_embeddings.get(word, np.array([-1.0, -1.0]))

    def compute_average(self, sentence, ignoreFiller=False):
        words = self._normalize(sentence)
        if ignoreFiller:
            words[:] = [w for w in words if w not in self.filler_words]

        total = np.array([0.0, 0.0])

        if (len(words) == 0):
            return total

        for w in words:
            total += self._get_embedding(w)

        return total / len(words)

    def compute_attention(self, sentence):
        words = self._normalize(sentence)
        total = np.array([0.0, 0.0])

        if (len(words) == 0):
            return total

        weighted_sum = [0.0, 0.0]
        total_weight = 0.0

        for w in words:
            emb = self._get_embedding(w)
            attn_w = self.attention_weights.get(w, 1.0)
            weighted_sum += (emb * attn_w)
            total_weight += attn_w

        return weighted_sum / total_weight


if __name__ == "__main__":
    embedder = SentenceEmbedder()
    example = "I love chocolate milk as well!"

    avg_emb = embedder.compute_average(example)
    avg_ign_filler = embedder.compute_average(example, ignoreFiller=True)
    attention_emb = embedder.compute_attention(example)

    print("Sentence:", example)
    print("Average embedding:", avg_emb)
    print("Average (ignoring filler) embedding:", avg_ign_filler)
    print("Attention-based embedding:", attention_emb)
