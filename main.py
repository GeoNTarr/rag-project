"""
Главный файл RAG системы

Требования:
- Python 3.8+
- Все зависимости из requirements.txt

Запуск:
1. Установить зависимости: pip install -r requirements.txt
2. Запустить: python main.py

Команда:
- Data Engineer: src/data/
- ML Developer: src/ml/
- NLP Engineer: src/nlp/
- Data Scientist: src/evaluation/
- Team Lead: интеграция и интерфейс
"""

import torch
import sys
import os
import yaml
from typing import Dict, Any, List
import time

# Добавляем src в путь для импортов
sys.path.append('src')


class ProjectIntegrator:
    """Интегратор всей RAG системы"""

    def __init__(self, config_path: str = "config.yaml"):
        self.device = self.setup_device()
        self.components: Dict[str, Any] = {}
        self.config = self.load_config(config_path)
        print(f"? Инициализация системы на устройстве: {self.device}")

    def setup_device(self) -> str:
        """Настраивает вычислительное устройство (GPU/CPU)"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"? GPU обнаружена: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        else:
            print("?? GPU не обнаружена, используем CPU")
            return "cpu"

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"?? Конфигурационный файл {config_path} не найден, используются значения по умолчанию")
            return {}
        except Exception as e:
            print(f"? Ошибка загрузки конфигурации: {e}")
            return {}

    def integrate_all_modules(self):
        """Интегрирует все модули от команды"""
        try:
            print("? Интеграция модулей...")

            # Импортируем модули команды
            from src.data.data_loader import DataLoader
            from src.ml.rag_system import RAGSystem
            from src.nlp.embeddings import EmbeddingModel
            from src.nlp.prompt_engine import PromptEngine
            from src.evaluation.tester import SystemTester

            # Инициализируем компоненты
            self.components['data_loader'] = DataLoader()
            self.components['embedding_model'] = EmbeddingModel(device=self.device)
            self.components['rag_system'] = RAGSystem(device=self.device)
            self.components['prompt_engine'] = PromptEngine(device=self.device)
            self.components['evaluator'] = SystemTester()

            # Настраиваем систему
            self.setup_rag_system()

            print("? Все модули успешно интегрированы")

        except ImportError as e:
            print(f"? Ошибка импорта модулей: {e}")
            print("Убедитесь, что все участники создали свои классы")
            return False
        except Exception as e:
            print(f"? Ошибка интеграции: {e}")
            return False

        return True

    def setup_rag_system(self):
        """Настраивает RAG систему с данными"""
        print("? Загрузка и подготовка данных...")

        # Получаем чанки от Data Engineer
        chunks = self.components['data_loader'].get_chunks()
        print(f"? Загружено {len(chunks)} чанков")

        # Строим векторный индекс
        self.components['rag_system'].build_index(
            chunks=chunks,
            embedding_model=self.components['embedding_model']
        )

        # Настраиваем порог релевантности
        threshold = self.config.get('model', {}).get('similarity_threshold', 0.6)
        self.components['rag_system'].set_threshold(threshold)

        print("? RAG система настроена и готова к работе")

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Обрабатывает вопрос через полный пайплайн

        Args:
            question: Вопрос пользователя

        Returns:
            Dict: Результат обработки
        """
        start_time = time.time()

        try:
            # 1. Проверка релевантности
            is_relevant = self.components['rag_system'].is_relevant(question)
            relevance_score = self.components['rag_system'].get_similarity_score(question)

            # 2. Поиск контекста и генерация ответа
            if is_relevant:
                context = self.components['rag_system'].search_similar(question)
                answer = self.components['prompt_engine'].generate_answer(context, question)
                response_type = "answer"
            else:
                answer = self.components['prompt_engine'].get_refusal_response(question)
                context = []
                response_type = "refusal"

            processing_time = time.time() - start_time

            return {
                "question": question,
                "answer": answer,
                "is_relevant": is_relevant,
                "relevance_score": relevance_score,
                "context": context,
                "response_type": response_type,
                "processing_time": processing_time,
                "success": True
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"Ошибка обработки: {str(e)}",
                "is_relevant": False,
                "relevance_score": 0.0,
                "context": [],
                "response_type": "error",
                "processing_time": time.time() - start_time,
                "success": False
            }

    def run_demo(self):
        """Запускает демонстрацию работы системы"""
        print("\n? Демонстрация RAG системы")
        print("=" * 50)

        # Тестовые вопросы для демонстрации
        demo_questions = [
            "Что такое Python?",
            "Объясните машинное обучение",
            "Сколько планет в солнечной системе?",
            "Как работает RAG?",
            "Кто президент России?"
        ]

        print("? Примеры вопросов для тестирования:")
        for i, question in enumerate(demo_questions, 1):
            print(f"  {i}. {question}")

        print("\n" + "=" * 50)

        while True:
            question = input("\n? Ваш вопрос (или 'выход' для завершения): ").strip()

            if question.lower() in ['выход', 'exit', 'quit']:
                break
            if not question:
                continue

            # Обработка вопроса
            result = self.process_question(question)

            # Вывод результата
            print(f"\n? Результат:")
            print(f"   ? Ответ: {result['answer']}")
            print(f"   ? Релевантность: {result['relevance_score']:.3f} ({result['is_relevant']})")
            print(f"   ?? Время обработки: {result['processing_time']:.2f} сек")

            if result['context']:
                print(f"   ? Использовано чанков: {len(result['context'])}")

    def run_evaluation(self):
        """Запускает оценку системы"""
        print("\n? Запуск оценки системы...")

        try:
            test_results = self.components['evaluator'].run_comprehensive_tests(self.components['rag_system'])

            print("? Результаты оценки:")
            for metric, value in test_results.items():
                print(f"   {metric}: {value}")

        except Exception as e:
            print(f"? Ошибка оценки: {e}")


def main():
    """Главная функция запуска системы"""
    print("=" * 60)
    print("? RAG System - Умная система вопросов и ответов")
    print("=" * 60)

    # Инициализация интегратора
    integrator = ProjectIntegrator()

    # Интеграция модулей
    success = integrator.integrate_all_modules()

    if not success:
        print("? Не удалось интегрировать модули. Завершение работы.")
        return

    # Запуск демо
    integrator.run_demo()

    # Оценка системы
    integrator.run_evaluation()

    print("\n? Завершение работы системы")


if __name__ == "__main__":
    main()





