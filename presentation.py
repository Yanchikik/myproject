import streamlit as st
import reveal_slides as rs
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import os

def presentation_page():
    st.title("Презентация проекта: Прогнозирование отказов оборудования")

    # Содержание презентации в формате Markdown
    presentation_markdown = """
    ## Прогнозирование отказов оборудования

    ### Введение
    - **Задача**: Предсказать вероятность отказа оборудования. Модель будет выдавать значение вероятности того, что оборудование выйдет из строя (отказ = 1) или останется работоспособным (отказ = 0).
    - **Датасет**: Используется синтетический датасет "AI4I 2020 Predictive Maintenance Dataset", содержащий 10 000 записей с 14 признаками.
    - **Цель**: Разработать модель машинного обучения для прогнозирования вероятности отказа оборудования.

    ---

    ### Этапы работы
    1. **Загрузка данных**:
        - Загрузка данных из CSV-файла или через библиотеку `ucimlrepo`.
        - Изучение структуры данных и их характеристик.

    2. **Предобработка данных**:
        - Удаление ненужных столбцов (например, уникальных идентификаторов).
        - Преобразование категориальных переменных в числовые.
        - Проверка на пропущенные значения и их обработка.
        - Масштабирование числовых признаков.

    3. **Обучение модели**:
        - Разделение данных на обучающую и тестовую выборки.
        - Обучение модели XGBoost для классификации (предсказание вероятности отказа оборудования).
        - Сравнение производительности моделей.

    4. **Оценка модели**:
        - Использование метрик: Accuracy, Confusion Matrix, ROC-AUC.
        - Визуализация результатов (графики ROC-кривых, матрицы ошибок).

    5. **Визуализация результатов**:
        - Построение графиков для анализа данных.
        - Отображение результатов предсказаний модели.

    ---

    ### Streamlit-приложение
    - **Основная страница**:
        - Загрузка данных.
        - Обучение модели.
        - Визуализация результатов.
        - Предсказание на новых данных.

    - **Страница с презентацией**:
        - Описание проекта.
        - Демонстрация этапов работы.

    ---

    ### Заключение
    - **Итоги**:
        - Модель успешно предсказывает вероятность отказа оборудования с высокой точностью.
        - Наилучшие результаты показала модель XGBoost.
    
    - **Возможные улучшения**:
        - Использование более сложных моделей (например, нейронные сети).
        - Увеличение объема данных для обучения.
        - Добавление новых признаков для улучшения предсказаний.
    """

    # Настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500, min_value=300, max_value=1000)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "latex", "mathjax2", "mathjax3", "notes", "search", "zoom"], default=["highlight", "notes"])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "\n---\n"},
    )

 # Запуск презентации
if __name__ == "__main__":
    presentation_page()
