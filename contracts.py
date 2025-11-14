# Contracts for RAG pipeline (AlfaBank case)
# --------------------------------------------
# Principles:
# - Free solutions and APIs only
# - Hit@5 metric
# - Top-5 documents for each question
# - Data: Questions.csv and Websites.csv

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Protocol
import pandas as pd
import numpy as np


# === 1 COMMON DATA STRUCTURES ===

@dataclass
class Question:
    q_id: str
    query: str

@dataclass
class Document:
    web_id: str
    url: str
    kind: str
    title: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    web_id: str
    text_chunk: str

@dataclass
class DataBundle:
    questions: pd.DataFrame
    websites: pd.DataFrame
    chunks: pd.DataFrame

@dataclass
class EmbeddingBundle:
    chunk_embeddings: np.ndarray
    question_embeddings: np.ndarray
    chunk_ids: List[str]
    question_ids: List[str]

@dataclass
class RetrievalResult:
    q_id: str
    top_docs: List[str]  # web_id of top-5 documents
    scores: List[float]  # relevance scores


# === 2 DATA CONTRACT (Data Engineer) ===

class DataContract(Protocol):
    """Contract for loading and preparing data"""

    def load_questions(self, file_path: str = "Questions.csv") -> pd.DataFrame:
        """Loads questions from CSV"""
        ...
    
    def load_websites(self, file_path: str = "Websites.csv") -> pd.DataFrame:
        """Loads web pages from CSV"""
        ...

    def preprocess_text(self, text: str) -> str:
        """Cleans and normalizes text (HTML tags, extra spaces, etc.)"""
        ...

    def chunk_documents(self, df: pd.DataFrame, chunk_size: int = 512, overlap: int = 50) -> pd.DataFrame:
        """Splits documents into overlapping chunks"""
        ...

    def build_data_bundle(self, questions_path: str = "Questions.csv", websites_path: str = "Websites.csv") -> DataBundle:
        """Creates complete data package for the pipeline"""
        ...

    def validate_data_quality(self) -> Dict[str, Any]:
        """Validates data quality (missing values, duplicates)"""
        ...


# === 3 EMBEDDING CONTRACT (NLP Engineer) ===

class EmbeddingContract(Protocol):
    """Contract for working with embeddings"""

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Creates embeddings for a list of texts"""
        ...

    def get_embedding_dimension(self) -> int:
        """Returns embedding dimensions"""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model information"""
        ...
    def get_free_embedding_model(self, model_path="./sentence-transformer") -> Any:
        """Returns free embedding model"""
        ...

    def get_free_reranker(self) -> Any:
        """Returns free reranker (optional for quality improvement)"""
        ...

    def get_available_models(self) -> List[str]:
        """Returns list of available free models"""
        ...


# === 4 RETRIEVAL CONTRACT (ML Developer) ===

class RetrievalContract(Protocol):
    """Contract for retrieval part (indexing + search)"""

    def build_vector_index(self, embeddings: np.ndarray, chunk_ids: List[str]) -> Any:
        """Builds vector index (FAISS/Annoy)"""
        ...

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Searches for top-K similar chunks (chunk_id, score)"""
        ...

    def retrieve_for_question(self, question: str, embedding_model: EmbeddingContract, top_k: int = 5) -> List[Tuple[str, float]]:
        """Performs search for a single question"""
        ...

    def batch_retrieve(self, questions: List[str], embedding_model: EmbeddingContract, top_k: int = 5) -> List[RetrievalResult]:
        """Performs search for a list of questions"""
        ...


# === 5 EVALUATION CONTRACT (Data Scientist) ===

class EvaluationContract(Protocol):
    """Contract for quality evaluation"""

    def calculate_hit_at_k(self, true_relevant: List[List[str]], predicted: List[List[str]], k: int = 5) -> float:
        """Calculates Hit@K for a set of questions"""
        ...

    def generate_submission_file(self, results: List[RetrievalResult], output_path: str = "submission.csv") -> pd.DataFrame:
        """Generates CSV file for submission"""
        ...

    def analyze_retrieval_quality(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Analyzes retrieval quality (score distribution, etc.)"""
        ...


# === 6 PIPELINE CONTRACT (Team Lead) ===

class PipelineContract(Protocol):
    """Contract for running the entire RAG pipeline"""

    def run_full_pipeline(self, questions_path: str = "Questions.csv", websites_path: str = "Websites.csv") -> Dict[str, Any]:
        """Runs the complete process: data -> embeddings -> index -> search -> evaluation"""
        ...

    def validate_data(self, questions_path: str, websites_path: str) -> bool:
        """Validates input data correctness"""
        ...

    def generate_final_submission(self) -> str:
        """Generates final submission file"""
        ...


