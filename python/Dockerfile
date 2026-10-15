# Reproducible runtime for itrialspace-eval.
#   docker build -t itrialspace-eval .
#   docker run --rm -v "$PWD/tests/fixtures:/data:ro" itrialspace-eval analyze /data/lesions_sample.csv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /work && chown app:app /work

USER app
WORKDIR /work
ENTRYPOINT ["itrialspace-eval"]
CMD ["--help"]
