"""
Контракты между модулями RAG системы

"""

from typing import List, Dict, Any, Tuple
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
        """
        pass

    @staticmethod
    def get_chunking_stats() -> Dict[str, Any]:
        """
        Возвращает статистику по чанкам

        Returns:
            Dict: Статистика чанкинга
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
    def get_embeddings(texts: List[str]) -> np.ndarray:
        """
        Создает эмбеддинги для текстов

        Args:
            texts: Список текстов для векторизации

        Returns:
            np.ndarray: Векторные представления текстов
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


class EvaluationContract:
    """Контракт Data Scientist"""

    @staticmethod
    def calculate_metrics(true_labels: List[bool], predictions: List[bool]) -> Dict[str, float]:
        """
        Вычисляет метрики качества системы

        Args:
            true_labels: Истинные метки
            predictions: Предсказания системы

        Returns:
            Dict: Метрики качества {'accuracy': 0.95, ...}
        """
        pass

    @staticmethod
    def run_comprehensive_tests(rag_system: Any) -> Dict[str, Any]:
        """
        Запускает комплексное тестирование системы

        Args:
            rag_system: Объект RAG системы

        Returns:
            Dict: Результаты тестирования
        """
        pass

    @staticmethod
    def find_optimal_threshold(test_questions: Dict[str, List[str]], rag_system: Any) -> float:
        """
        Находит оптимальный порог для детектора незнания

        Args:
            test_questions: Тестовые вопросы
            rag_system: Объект RAG системы

        Returns:
            float: Оптимальный порог
        """
        pass