# backend/Dockerfile
FROM python:3.8

ADD requirements.txt .
#WORKDIR /lead_converision

RUN pip install -r requirements.txt 

#COPY . /lead_converision

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["streamlit_app.py"]