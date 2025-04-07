import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

def analysis_and_model_page():
    st.title("Анализ данных и модель (XGBoost)")

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

        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        scaler = StandardScaler()
        numerical_features = ['Air temperature', 'Process temperature', 
                            'Rotational speed', 'Torque', 'Tool wear']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Обучение XGBoost
        st.header("Обучение модели XGBoost")
        
        with st.expander("Настройки модели"):
            n_estimators = st.slider("Количество деревьев", 50, 500, 100)
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
            max_depth = st.slider("Максимальная глубина", 3, 10, 6)
            
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Учет дисбаланса классов
        )
        
        with st.spinner('Модель обучается...'):
            model.fit(X_train, y_train)
            st.success("Обучение завершено!")

        # Оценка модели
        st.header("Результаты обучения")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
        
        with col2:
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_title('ROC Curve')
        ax2.legend()
        
        st.pyplot(fig)

        # Важность признаков
        st.subheader("Важность признаков")
        importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': model.feature_importances_
        }).sort_values('Важность', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Важность', y='Признак', data=importance, ax=ax)
        st.pyplot(fig)

        # Прогнозирование
        st.header("Прогнозирование отказов")
        with st.form("prediction_form"):
            st.write("Введите параметры оборудования:")
            
            col1, col2 = st.columns(2)
            with col1:
                product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
                air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                process_temp = st.number_input("Температура процесса [K]", value=310.0)
            with col2:
                rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]", value=0)
            
            if st.form_submit_button("Сделать прогноз"):
                input_data = pd.DataFrame({
                    'Type': [0 if product_type == "L" else 1 if product_type == "M" else 2],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })
                
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                
                st.success(f"Прогноз: {'⚠️ Отказ оборудования' if prediction[0] == 1 else '✅ Нормальная работа'}")
                st.metric("Вероятность отказа", f"{prediction_proba[0]:.2%}")
                
                if prediction[0] == 1:
                    st.warning("Рекомендуется провести техническое обслуживание!")
                else:
                    st.info("Оборудование работает в нормальном режиме")

if __name__ == "__main__":
    analysis_and_model_page()