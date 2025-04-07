import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # Измененный импорт
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    st.header("Загрузка данных")
    data = None

    try:
        # Попытка загрузить данные через ucimlrepo
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        st.success("Данные успешно загружены через ucimlrepo.")
    except Exception as e:
        st.warning(f"Не удалось загрузить данные через ucimlrepo. Ошибка: {e}")
        st.info("Попытка загрузить данные из локального CSV-файла...")

      # 2. Попытка загрузить из локального файла
        try:
            # Путь относительно расположения скрипта
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
        # Удаление ненужных столбцов
        columns_to_drop = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        data = data.drop(columns=columns_to_drop)

        # Преобразование категориальной переменной Type в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование числовых признаков
        scaler = StandardScaler()
        numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели (ЗАМЕНА НА XGBoost)
        st.header("Обучение модели")
        model = XGBClassifier(random_state=42)  # Измененная строка
        model.fit(X_train, y_train)

        # Оценка модели
        st.header("Результаты обучения модели")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
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
                # Преобразование введенных данных
                input_data = pd.DataFrame({
                    'Type': [0 if product_type == "L" else 1 if product_type == "M" else 2],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })

                # Масштабирование данных
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                # Предсказание
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]

                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")

# Запуск страницы
if __name__ == "__main__":
    analysis_and_model_page()