# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта
Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0). Результаты работы оформлены в виде интерактивного Streamlit-приложения.

## 📊 Датасет
Используется датасет **[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data)**:
- 10 000 записей
- 14 признаков
- Целевая переменная: Machine failure (0/1)

## ⚙️ Установка и запуск
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Yanchikik/myproject.git
   cd myproject
2. Установите зависимости: 
   pip install -r requirements.txt 
3. Запустите приложение: 
   streamlit run app.py 
## Структура репозитория - `app.py`: Основной файл приложения. - `analysis_and_model.py`: Страница с анализом данных и моделью. - `presentation.py`: Страница с презентацией проекта. - `requirements.txt`: Файл с зависимостями. - `data/`: Папка с данными. - `README.md`: Описание проекта. 
## Видео-демонстрация 
[Ссылка на видео](video/demo.mp4) или встроенное видео ниже: 
<video src="video/demo.mp4" controls width="100%"></video> 