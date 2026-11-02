FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ src/
COPY app/ app/
COPY models/dengue_outbreak_model.pkl models/
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
