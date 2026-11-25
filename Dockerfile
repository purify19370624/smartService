FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libmagic1：unstructured 的文件类型识别；其余为 docx/pdf 解析的最小运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install "setuptools<81" && pip install -r requirements.txt

COPY . .

# 运行时数据目录（向量库 / 会话库 / 日志），docker run 时可挂载持久化
RUN mkdir -p chroma/knowledge db logs
VOLUME ["/app/chroma", "/app/db", "/app/logs"]

EXPOSE 7860

# 启动预热点接触发向量检索，等 Gradio 真正可访问再算健康
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

CMD ["python", "main.py"]
