FROM python:3.7.8
# Setup workdir
WORKDIR /usr/src/app
# Copy files
COPY requirements.txt ./
COPY requirements_http.txt ./
COPY . .
# Dependency Installation
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements_http.txt
# Download Models
RUN (cd tools && python3 setup.py all)
# Setting environment variables
ENV HOST=0.0.0.0
ENV PORT=5000
ENV AUTH=False
ENV MODEL=u2net
ENV PREPROCESSING=none
ENV POSTPROCESSING=fba
ENV ADMIN_TOKEN=admin
ENV ALLOWED_TOKENS_PYTHON_ARR=["test"]
ENV IS_DOCKER_CONTAINER=True

EXPOSE 5000

ENTRYPOINT ["python3", "./http_api.py"]
