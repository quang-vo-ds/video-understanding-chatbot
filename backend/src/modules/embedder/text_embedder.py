from milvus_model.hybrid import BGEM3EmbeddingFunction


class TextEmbedder:
    def __init__(self) -> None:
        super().__init__()
        self.embeddor = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")

    @property
    def dim(self):
        return self.embeddor.dim["dense"]

    def encode(self, inp: list[str]) -> list[list[float]]:
        emb_results = self.embeddor(inp)

        dense_vectors = []
        for text in inp:
            if not text:
                dense_vectors.append([0.0 for _ in range(self.dim)])
            else:
                emb_results = self.embeddor([text])
                dense_vectors.append(emb_results["dense"][0])

        return dense_vectors
        
