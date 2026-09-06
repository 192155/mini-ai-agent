import os
import pickle
import numpy as np


class VectorStore:

    def __init__(self):

        self.documents = []

        self.embeddings = None

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

        embeddings = np.array(
            embeddings
        )

        if self.embeddings is None:

            self.embeddings = embeddings

        else:

            self.embeddings = np.vstack(
                [
                    self.embeddings,
                    embeddings
                ]
            )

    def search(
        self,
        query_embedding,
        top_k=5,
        threshold=0.30
    ):

        if (
            self.embeddings is None
            or
            len(self.documents) == 0
        ):

            return []

        query_embedding = np.array(
            query_embedding
        )

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        valid_indices = np.where(
            scores >= threshold
        )[0]

        if len(valid_indices) == 0:
            return []

        sorted_indices = valid_indices[
            np.argsort(
                scores[valid_indices]
            )[::-1]
        ]

        top_indices = sorted_indices[
            :top_k
        ]

        results = []

        for index in top_indices:

            results.append(
                {
                    "text": self.documents[index],
                    "score": float(
                        scores[index]
                    )
                }
            )

        return results

    def save(self, path):

        directory = os.path.dirname(path)

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

    def load(self, path):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Vector store not found: {path}"
            )

        with open(
            path,
            "rb"
        ) as file:

            data = pickle.load(file)

        self.documents = data.get(
            "documents",
            []
        )

        self.embeddings = data.get(
            "embeddings",
            None
        )

        print(
            "Vector store loaded successfully."
        )