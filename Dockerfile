FROM mambaorg/micromamba:2.9.0

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create --yes --file /tmp/environment.yml && micromamba clean --all --yes

ENV ENV_NAME=ptk_312
ENV PYTHONPATH=/app

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER option_stream/ /app/option_stream/

EXPOSE 5006
CMD ["bokeh", "serve", "option_stream/main.py", "--address", "0.0.0.0", "--port", "5006", "--allow-websocket-origin", "localhost:5006", "--allow-websocket-origin", "127.0.0.1:5006"]
