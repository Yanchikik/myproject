import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
import os

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    st.header("Загрузка данных")
    data = None

    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        st.success("Данные успешно загружены через ucimlrepo.")
    except Exception as e:
        st.warning(f"Не удалось загрузить данные через ucimlrepo. Ошибка: {e}")
        st.info("Попытка загрузить данные из локального CSV-файла...")

        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            csv_path = os.path.join(data_dir, 'ai4i2020.csv')

            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                st.success(f"Данные успешно загружены из локального файла: {csv_path}")
            else:
                st.error(f"Файл не найден по пути: {csv_path}")
                st.info("""
                Для работы приложения необходимо:
                1. Создать папку 'data' в той же директории, где находится app.py
                2. Поместить туда файл 'ai4i2020.csv'
                """)
                st.stop()

        except Exception as e:
            st.error(f"Ошибка при загрузке локального файла: {e}")
            st.stop()

    # Предобработка данных
    st.header("Предобработка данных")
    if data is not None:
        columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        data = data.drop(columns=columns_to_drop)

        # Преобразование категориальной переменной Type в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Переименование признаков (удаление скобок и пробелов)
        data.columns = data.columns.str.replace(r"[\[\]<>]", "", regex=True)
        data.columns = data.columns.str.replace(" ", "_")

        # Обновлённые названия признаков
        numerical_features = [
            'Air_temperature_K',
            'Process_temperature_K',
            'Rotational_speed_rpm',
            'Torque_Nm',
            'Tool_wear_min'
        ]

        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        X = data.drop(columns=['Machine_failure'])
        y = data['Machine_failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели XGBoost
        st.header("Обучение модели (XGBoost)")
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Оценка модели
        st.header("Результаты обучения модели")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Матрица ошибок")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Отчёт о классификации")
        st.text(classification_rep)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            product_type = st.selectbox("Тип продукта (Type)", ["L", "M", "H"])
            air_temp = st.number_input("Температура окружающей среды [K]")
            process_temp = st.number_input("Рабочая температура [K]")
            rotational_speed = st.number_input("Скорость вращения [rpm]")
            torque = st.number_input("Крутящий момент [Nm]")
            tool_wear = st.number_input("Износ инструмента [min]")
            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                input_data = pd.DataFrame({
                    'Type': [0 if product_type == "L" else 1 if product_type == "M" else 2],
                    'Air_temperature_K': [air_temp],
                    'Process_temperature_K': [process_temp],
                    'Rotational_speed_rpm': [rotational_speed],
                    'Torque_Nm': [torque],
                    'Tool_wear_min': [tool_wear]
                })

                input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]

                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")
