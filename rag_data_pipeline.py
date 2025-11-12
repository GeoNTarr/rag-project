import pandas as pd
import numpy as np
import hashlib
from typing import Dict, Any, List
from data_contracts import DataContract, DataBundle, Document, Question, Chunk


class AlfabankDataProcessor(DataContract):
    def __init__(self, data_path: str = "."):
        self.data_path = data_path

    def load_questions(self, file_path: str = "questions_clean.csv") -> pd.DataFrame:
        try:
            if self.data_path == ".":
                full_path = file_path
            else:
                full_path = f"{self.data_path}/{file_path}"

            df = pd.read_csv(full_path)
            print(f" questions_clean загружено: {df.shape}")
            return df
        except FileNotFoundError:
            print(f" Файл {file_path} не найден")
            return pd.DataFrame()
        except Exception as e:
            print(f" Ошибка загрузки {file_path}: {e}")
            return pd.DataFrame()

    def load_websites(self, file_path: str = "websites_updated.csv") -> pd.DataFrame:
        try:
            if self.data_path == ".":
                full_path = file_path
            else:
                full_path = f"{self.data_path}/{file_path}"

            df = pd.read_csv(full_path)
            print(f" websites_updated загружено: {df.shape}")
            return df
        except FileNotFoundError:
            print(f" Файл {file_path} не найден")
            return pd.DataFrame()
        except Exception as e:
            print(f" Ошибка загрузки {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text_str = str(text)
        # Базовая очистка
        text_clean = text_str.strip()
        return text_clean

    def chunk_documents(self, df: pd.DataFrame, chunk_size: int = 512, overlap: int = 50) -> pd.DataFrame:
        if df.empty or 'text' not in df.columns:
            print(" Нет данных для чанкинга")
            return pd.DataFrame()

        chunks_data = []
        chunk_counter = 0

        for idx, row in df.iterrows():
            text_content = str(row.get('text', ''))
            text_length = len(text_content)
            web_id = row.get('web_id', f'doc_{idx}')

            if text_length == 0:
                continue

            # Простой чанкинг по символам
            if text_length <= chunk_size:
                chunks = [text_content]
            else:
                chunks = []
                start = 0
                while start < text_length:
                    end = start + chunk_size
                    chunk = text_content[start:end]
                    chunks.append(chunk)
                    start += chunk_size - overlap
                    if start >= text_length:
                        break

            # Создаем чанки
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{web_id}_chunk_{i}"
                chunks_data.append({
                    'chunk_id': chunk_id,
                    'web_id': web_id,
                    'text_chunk': chunk_text,
                    'chunk_length': len(chunk_text),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                chunk_counter += 1

        print(f" Создано чанков: {chunk_counter}")
        return pd.DataFrame(chunks_data)

    def build_data_bundle(self, questions_path: str = "questions_clean.csv",
                          websites_path: str = "websites_updated.csv") -> DataBundle:
        print(" Создание DataBundle для RAG системы")

        # Загрузка данных
        questions_df = self.load_questions(questions_path)
        websites_df = self.load_websites(websites_path)

        if questions_df.empty or websites_df.empty:
            print(" Не удалось загрузить данные")
            return DataBundle(
                questions=pd.DataFrame(),
                websites=pd.DataFrame(),
                chunks=pd.DataFrame()
            )

        # Базовая предобработка
        questions_df = self._preprocess_questions(questions_df)
        websites_df = self._preprocess_websites(websites_df)

        # Создание чанков
        chunks_df = self.chunk_documents(websites_df)

        print(f" DataBundle создан:")
        print(f"   - Вопросов: {len(questions_df)}")
        print(f"   - Документов: {len(websites_df)}")
        print(f"   - Чанков: {len(chunks_df)}")

        return DataBundle(
            questions=questions_df,
            websites=websites_df,
            chunks=chunks_df
        )

    def validate_data_quality(self) -> Dict[str, Any]:
        print("\n ВАЛИДАЦИЯ КАЧЕСТВА ДАННЫХ")
        print("=" * 50)

        quality_report = {}

        # Загрузка данных для валидации
        questions_df = self.load_questions()
        websites_df = self.load_websites()

        # Валидация questions_clean
        if not questions_df.empty:
            q_issues = self._validate_questions(questions_df)
            quality_report['questions'] = q_issues

        # Валидация websites_updated
        if not websites_df.empty:
            w_issues = self._validate_websites(websites_df)
            quality_report['websites'] = w_issues

        # Общий отчет
        total_issues = sum(len(issues) for issues in quality_report.values())
        quality_report['summary'] = {
            'total_issues': total_issues,
            'status': 'PASS' if total_issues == 0 else 'WARNING'
        }

        print(f" ИТОГИ ВАЛИДАЦИИ: {quality_report['summary']['status']}")
        print(f"   - Найдено проблем: {total_issues}")

        return quality_report

    def _preprocess_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # Заполнение пропусков
        df_clean = df_clean.fillna('')

        # Очистка текстовых колонок
        if 'query' in df_clean.columns:
            df_clean['query'] = df_clean['query'].apply(self.preprocess_text)

        # Удаление дубликатов
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)

        if initial_rows != final_rows:
            print(f"   - Удалено дубликатов вопросов: {initial_rows - final_rows}")

        return df_clean

    def _preprocess_websites(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # Заполнение пропусков
        df_clean = df_clean.fillna('')

        # Очистка текстовых колонок
        text_columns = ['text', 'title', 'url', 'kind']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.preprocess_text)

        # Удаление дубликатов
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)

        if initial_rows != final_rows:
            print(f"   - Удалено дубликатов документов: {initial_rows - final_rows}")

        return df_clean

    def _validate_questions(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}

        # Проверка обязательных колонок
        required_columns = ['q_id', 'query']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues['missing_columns'] = missing_columns

        # Проверка пропущенных значений
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            issues['missing_values'] = missing_values

        # Проверка дубликатов
        duplicate_q_ids = df['q_id'].duplicated().sum() if 'q_id' in df.columns else 0
        duplicate_queries = df['query'].duplicated().sum() if 'query' in df.columns else 0

        if duplicate_q_ids > 0:
            issues['duplicate_q_ids'] = duplicate_q_ids
        if duplicate_queries > 0:
            issues['duplicate_queries'] = duplicate_queries

        # Вывод результатов
        print(f" Валидация questions_clean:")
        print(f"   - Пропущенные значения: {sum(missing_values.values())}")
        print(f"   - Дубликаты q_id: {duplicate_q_ids}")
        print(f"   - Дубликаты query: {duplicate_queries}")

        return issues

    def _validate_websites(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}

        # Проверка обязательных колонок
        required_columns = ['web_id', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues['missing_columns'] = missing_columns

        # Проверка пропущенных значений
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            issues['missing_values'] = missing_values

        # Проверка дубликатов
        duplicate_web_ids = df['web_id'].duplicated().sum() if 'web_id' in df.columns else 0

        # Проверка дубликатов контента через хеши
        if 'text' in df.columns:
            df_temp = df.copy()
            df_temp['text_hash'] = df_temp['text'].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest()
            )
            duplicate_texts = df_temp['text_hash'].duplicated().sum()
            if duplicate_texts > 0:
                issues['duplicate_texts'] = duplicate_texts

        if duplicate_web_ids > 0:
            issues['duplicate_web_ids'] = duplicate_web_ids

        # Анализ длины текста
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            empty_texts = (text_lengths == 0).sum()
            if empty_texts > 0:
                issues['empty_texts'] = empty_texts

            print(f" Валидация websites_updated:")
            print(f"   - Пропущенные значения: {sum(missing_values.values())}")
            print(f"   - Дубликаты web_id: {duplicate_web_ids}")
            print(f"   - Дубликаты текста: {issues.get('duplicate_texts', 0)}")
            print(f"   - Пустые тексты: {empty_texts}")
            print(f"   - Средняя длина текста: {text_lengths.mean():.1f} символов")

        return issues