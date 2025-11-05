"""
Контракты между модулями RAG системы

Каждый участник должен реализовать эти интерфейсы в своих классах
"""

from typing import List, Dict, Any, Union
import numpy as np


class DataContract:
    """Контракт Data Engineer с командой"""

    @staticmethod
    def get_chunks() -> List[str]:
        """
        Возвращает список текстовых чанков из базы знаний

        Returns:
            List[str]: Список текстовых фрагментов для поиска
        """
        pass

    @staticmethod
    def get_test_questions() -> Dict[str, List[str]]:
        """
        Возвращает тестовые вопросы для оценки системы

        Returns:
            Dict: {'in_scope': [], 'out_scope': []}
                  in_scope - вопросы из базы знаний
                  out_scope - вопросы вне базы знаний
        """
        pass

    @staticmethod
    def get_chunking_stats() -> Dict[str, Any]:
        """
        Возвращает статистику по чанкам

        Returns:
            Dict: {
                'total_chunks': int,
                'avg_chunk_length': float,
                'min_chunk_length': int,
                'max_chunk_length': int,
                'total_words': int
            }
        """
        pass


class MLContract:
    """Контракт ML-разработчика"""

    @staticmethod
    def is_relevant(question: str, threshold: float = 0.6) -> bool:
        """
        Определяет, относится ли вопрос к базе знаний

        Args:
            question: Вопрос пользователя
            threshold: Порог схожести (0.0-1.0)

        Returns:
            bool: True если вопрос релевантен базе знаний
        """
        pass

    @staticmethod
    def search_similar(question: str, top_k: int = 3) -> List[str]:
        """
        Ищет похожие чанки в базе знаний

        Args:
            question: Вопрос пользователя
            top_k: Количество возвращаемых чанков

        Returns:
            List[str]: Релевантные текстовые чанки
        """
        pass

    @staticmethod
    def get_similarity_score(question: str) -> float:
        """
        Возвращает оценку схожести вопроса с базой знаний

        Args:
            question: Вопрос пользователя

        Returns:
            float: Оценка схожести (0.0-1.0)
        """
        pass

    @staticmethod
    def build_index(chunks: List[str], embedding_model: Any):
        """
        Строит векторный индекс из чанков

        Args:
            chunks: Список текстовых чанков
            embedding_model: Модель для создания эмбеддингов
        """
        pass

    @staticmethod
    def set_threshold(threshold: float):
        """
        Устанавливает порог релевантности

        Args:
            threshold: Новый порог (0.0-1.0)
        """
        pass


class NLPContract:
    """Контракт NLP-инженера"""

    @staticmethod
    def generate_answer(context: List[str], question: str) -> str:
        """
        Генерирует ответ на основе контекста

        Args:
            context: Релевантные чанки из базы знаний
            question: Вопрос пользователя

        Returns:
            str: Сгенерированный ответ
        """
        pass

    @staticmethod
    def get_embeddings(texts: List[str]) -> Union[np.ndarray, Any]:
        """
        Создает эмбеддинги для текстов

        Args:
            texts: Список текстов для векторизации

        Returns:
            np.ndarray или torch.Tensor: Векторные представления текстов
        """
        pass

    @staticmethod
    def get_refusal_response(question: str) -> str:
        """
        Генерирует вежливый отказ для вопросов вне базы знаний

        Args:
            question: Вопрос пользователя

        Returns:
            str: Текст отказа
        """
        pass

    @staticmethod
    def get_embedding_dimension() -> int:
        """
        Возвращает размерность эмбеддингов

        Returns:
            int: Размерность векторных представлений
        """
        pass


class EvaluationContract:
    """Контракт Data Scientist"""

    @staticmethod
    def calculate_metrics(true_labels: List[bool], predictions: List[bool]) -> Dict[str, float]:
        """
        Вычисляет метрики качества системы

        Args:
            true_labels: Истинные метки (True/False)
            predictions: Предсказания системы (True/False)

        Returns:
            Dict: {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'specificity': float
            }
        """
        pass

    @staticmethod
    def run_comprehensive_tests(rag_system: Any) -> Dict[str, Any]:
        """
        Запускает комплексное тестирование системы

        Args:
            rag_system: Объект RAG системы

        Returns:
            Dict: Результаты тестирования {
                'performance': Dict,
                'accuracy': Dict,
                'edge_cases': Dict,
                'summary': Dict
            }
        """
        pass

    @staticmethod
    def find_optimal_threshold(test_questions: Dict[str, List[str]], rag_system: Any) -> float:
        """
        Находит оптимальный порог для детектора незнания

        Args:
            test_questions: {'in_scope': [], 'out_scope': []}
            rag_system: Объект RAG системы

        Returns:
            float: Оптимальный порог
        """
        pass

    @staticmethod
    def calculate_response_time_metrics(response_times: List[float]) -> Dict[str, float]:
        """
        Рассчитывает метрики времени ответа

        Args:
            response_times: Список времен ответа в секундах

        Returns:
            Dict: {
                'mean_response_time': float,
                'median_response_time': float,
                'p95_response_time': float
            }
        """
        pass


class IntegrationContract:
    """Контракт Team Lead (для интеграции)"""

    @staticmethod
    def integrate_all_modules() -> bool:
        """
        Интегрирует все модули от команды

        Returns:
            bool: True если интеграция успешна
        """
        pass

    @staticmethod
    def process_question(question: str) -> Dict[str, Any]:
        """
        Обрабатывает вопрос через полный пайплайн

        Args:
            question: Вопрос пользователя

        Returns:
            Dict: {
                'question': str,
                'answer': str,
                'is_relevant': bool,
                'relevance_score': float,
                'context': List[str],
                'response_type': str,
                'processing_time': float,
                'success': bool
            }
        """
        pass

    @staticmethod
    def setup_rag_system():
        """
        Настраивает RAG систему с данными
        """
        pass