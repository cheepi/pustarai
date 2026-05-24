FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /pustarai

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p pustara_models || true && \
    mv *.pkl metadata.json pustara_models/ 2>/dev/null || true

EXPOSE 8001

CMD ["uvicorn", "pustarai.fastapi_server:app", "--host", "0.0.0.0", "--port", "8001"]