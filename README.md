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
2. Установка зависимостей  
Рекомендуется использовать виртуальное окружение для изоляции зависимостей.  

🖥 Для Windows  
```sh
# Создаем виртуальное окружение  
python -m venv venv  

# Активируем его  
venv\Scripts\activate  

# Устанавливаем зависимости  
pip install -r requirements.txt  

# Если возникает ошибка доступа  
pip install --user -r requirements.txt  
```

💻 Для Linux / macOS  
```sh
# Убедимся, что venv установлен  
sudo apt install python3-venv  

# Создаем виртуальное окружение  
python3 -m venv venv  

# Активируем окружение  
source venv/bin/activate  

# Устанавливаем зависимости  
pip install -r requirements.txt  
```

 

3. Запустите приложение: 
   streamlit run app.py 
## 📁 Структура репозитория

- **`app.py`** – основной файл приложения Streamlit.  
- **`analysis_and_model.py`** – страница с анализом данных и моделью машинного обучения.  
- **`presentation.py`** – страница с презентацией проекта.  
- **`requirements.txt`** – файл со списком зависимостей для установки.  
- **`data/ai4i2020.csv`** – папка с данными, используемыми для анализа и обучения модели.  
- **`README.md`** – описание проекта, инструкции по установке и использованию.  

## 🎥 Видео-демонстрация  

[Скачать видео](https://github.com/Yanchikik/myproject/raw/master/video.mp4)
[Смотреть видео](https://Yanchikik.github.io/myproject/video/video.mp4)
<video src="https://Yanchikik.github.io/myproject/video/demo.mp4" controls width="100%"></video>