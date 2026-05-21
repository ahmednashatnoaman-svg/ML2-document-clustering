FROM python:3.11-slim

WORKDIR /app

# System deps for scipy / matplotlib / gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK resources so the container works offline
RUN python -c "\
import nltk; \
[nltk.download(r, quiet=True) for r in \
 ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']]"

COPY . .

EXPOSE 8501 8000

# Default: Streamlit dashboard
# Override CMD to run the API: docker run ... uvicorn app.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
