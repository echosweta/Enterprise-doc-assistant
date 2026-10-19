FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from eda.pipeline import get_index; get_index(rebuild=True)"
EXPOSE 8000
CMD ["uvicorn", "eda.api:app", "--host", "0.0.0.0", "--port", "8000"]
