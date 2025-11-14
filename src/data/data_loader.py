from typing import List
import os
from sentence_transformers import SentenceTransformer
from contracts import EmbeddingContract

class DataLoader(EmbeddingContract):
    # Предполагается, что папка с моделью называется sentence-transformer и расположена в папке проекта
    def get_free_embedding_model(self, model_path="./sentence-transformer"):
        return SentenceTransformer(model_path)

    def get_available_models(self) -> List[str]:
        return ["ai-forever/sbert_large_nlu_ru"]




