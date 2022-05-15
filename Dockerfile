# backend/Dockerfile

FROM python:3.8

COPY requirements.txt .
#WORKDIR /lead_converision

RUN pip install -r requirements.txt

#COPY . /lead_converision

EXPOSE 8000:8000

CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000" , "--reload"]