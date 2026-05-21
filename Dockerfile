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

EXPOSE 7860

# HuggingFace Spaces uses port 7860; local docker can override with -p 8501:7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
