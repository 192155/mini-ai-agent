from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        print(
            "Loading embedding model..."
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "Embedding model loaded successfully."
        )

    def create_embedding(self, text):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding