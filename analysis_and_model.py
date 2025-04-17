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
import re


def analysis_and_model_page():
    st.title("Анализ данных и модель предсказания отказа")

    # --- Загрузка данных ---
    st.header("1. Загрузка данных")
    data = None
    loaded_from_ucirepo = False

    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        loaded_from_ucirepo = True
        st.success("Данные успешно загружены через UCI ML Repo.")
    except Exception as e:
        st.warning(f"Не удалось загрузить через UCI ML Repo. Ошибка: {e}")
        st.info("Попытка загрузить локально...")

        try:
            data_path = os.path.join(os.path.dirname(__file__), "data", "ai4i2020.csv")
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                st.success(f"Загружено из локального файла: {data_path}")
            else:
                st.error(f"Файл не найден: {data_path}")
                st.stop()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            st.stop()

    # --- Предобработка ---
    st.header("2. Предобработка данных")

    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

    if 'Type' in data.columns:
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

    data.columns = data.columns.str.strip().str.replace(r"[^\w]", "_", regex=True)

    def find_col(columns, *patterns):
        for pat in patterns:
            match = [c for c in columns if re.search(pat, c, re.IGNORECASE)]
            if match:
                return match[0]
        return None

    air_temp_col = find_col(data.columns, "air.*temp")
    proc_temp_col = find_col(data.columns, "process.*temp")
    speed_col = find_col(data.columns, "rotational.*speed")
    torque_col = find_col(data.columns, "torque")
    wear_col = find_col(data.columns, "tool.*wear")

    numerical_features = [air_temp_col, proc_temp_col, speed_col, torque_col, wear_col]
    numerical_features = [f for f in numerical_features if f]

    if len(numerical_features) < 5:
        st.error("Не удалось автоматически определить все числовые признаки.")
        st.stop()

    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # --- Разделение данных ---
    X = data.drop(columns=['Machine_failure'])
    y = data['Machine_failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Обучение модели ---
    st.header("3. Обучение модели (XGBoost)")
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # --- Оценка модели ---
    st.header("4. Результаты модели")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    st.markdown(f"Точность (accuracy): {accuracy:.2f}")
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Отчет о классификации")
    st.text(class_report)

    # --- Предсказание ---
    st.header("Предсказание по новым данным")
    with st.form("prediction_form"):
        st.write("Введите значения признаков для предсказания:")
        productID = st.selectbox("Тип продукта (productID)", ["L", "M", "H"])
        air_temp = st.number_input("Температура окружающей среды [K]")
        process_temp = st.number_input("Рабочая температура [K]")
        rotational_speed = st.number_input("Скорость вращения [rpm]")
        torque = st.number_input("Крутящий момент [Nm]")
        tool_wear = st.number_input("Износ инструмента [min]")
        submit_button = st.form_submit_button("Предсказать")

        if submit_button:
            input_data = pd.DataFrame({
                'Type': [0 if productID == "L" else 1 if productID == "M" else 2],
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


