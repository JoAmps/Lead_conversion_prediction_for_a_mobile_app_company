FROM python:3.8

WORKDIR /lead_conversion-app

COPY . /usr/app/
EXPOSE 8000
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","train_model.py"]