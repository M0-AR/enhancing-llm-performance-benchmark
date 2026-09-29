FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PIP_NO_CACHE_DIR=1
WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 --prefer-binary -r /work/requirements.txt
COPY . /work
CMD ["python", "scripts/run_all.py", "--out", "results/universe_report.json"]
