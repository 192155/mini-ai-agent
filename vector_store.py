import os
import pickle
import math


class VectorStore:

    def __init__(self):

        self.documents = []

        self.embeddings = []

    def add_documents(
        self,
        documents,
        embeddings
    ):

        if not documents:
            return

        self.documents.extend(
            documents
        )

        for embedding in embeddings:

            self.embeddings.append(
                list(embedding)
            )

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):

        if (
            not vector_a
            or not vector_b
        ):
            return 0.0

        length = min(
            len(vector_a),
            len(vector_b)
        )

        dot_product = 0.0

        magnitude_a = 0.0

        magnitude_b = 0.0

        for i in range(length):

            a = float(
                vector_a[i]
            )

            b = float(
                vector_b[i]
            )

            dot_product += (
                a * b
            )

            magnitude_a += (
                a * a
            )

            magnitude_b += (
                b * b
            )

        magnitude_a = math.sqrt(
            magnitude_a
        )

        magnitude_b = math.sqrt(
            magnitude_b
        )

        if (
            magnitude_a == 0
            or
            magnitude_b == 0
        ):

            return 0.0

        return (
            dot_product
            /
            (
                magnitude_a
                *
                magnitude_b
            )
        )

    def search(
        self,
        query_embedding,
        top_k=5,
        threshold=0.30
    ):

        if (
            not self.embeddings
            or
            not self.documents
        ):

            return []

        scores = []

        for index, embedding in enumerate(
            self.embeddings
        ):

            similarity = (
                self.cosine_similarity(
                    query_embedding,
                    embedding
                )
            )

            if similarity >= threshold:

                scores.append(
                    (
                        similarity,
                        index
                    )
                )

        if not scores:

            return []

        scores.sort(
            key=lambda x: x[0],
            reverse=True
        )

        top_results = scores[
            :top_k
        ]

        results = []

        for score, index in top_results:

            document = self.documents[
                index
            ]

            # Supports both:
            # plain string documents
            # and {"text": ..., "source": ...}
            if isinstance(
                document,
                dict
            ):

                text = document.get(
                    "text",
                    ""
                )

                source = document.get(
                    "source",
                    "unknown"
                )

                results.append(
                    {
                        "text": text,
                        "source": source,
                        "score": float(score)
                    }
                )

            else:

                results.append(
                    {
                        "text": document,
                        "source": "unknown",
                        "score": float(score)
                    }
                )

        return results

    def save(
        self,
        path
    ):

        directory = os.path.dirname(
            path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }

        with open(
            path,
            "wb"
        ) as file:

            pickle.dump(
                data,
                file
            )

        print(
            "Vector store saved successfully."
        )

    def load(
        self,
        path
    ):

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"Vector store not found: {path}"
            )

        with open(
            path,
            "rb"
        ) as file:

            data = pickle.load(
                file
            )

        self.documents = data.get(
            "documents",
            []
        )

        loaded_embeddings = data.get(
            "embeddings",
            []
        )

        if loaded_embeddings is None:

            self.embeddings = []

        else:

            self.embeddings = [
                list(embedding)
                for embedding
                in loaded_embeddings
            ]

        print(
            "Vector store loaded successfully."
        )