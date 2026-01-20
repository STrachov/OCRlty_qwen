FROM vllm/vllm-openai:v0.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tools for git sync + readiness checks
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      openssh-client \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Orchestrator deps (vLLM already included in base image)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# Fallback baked-in code (optional but useful)
# If you really want "only git code", you can remove this COPY.
COPY app /opt/app/app

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8080
ENTRYPOINT ["/entrypoint.sh"]
