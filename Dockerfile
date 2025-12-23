# Используем легкий образ Python
FROM python:3.9-slim

# Рабочая директория
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Указываем порт (Yandex передает его через переменную окружения PORT)
ENV PORT 8080

# Команда запуска сервера через Gunicorn (для продакшена)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app