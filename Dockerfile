# backend/Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
COPY /app/requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY ./functions /functions/
COPY ./app /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--reload"]

