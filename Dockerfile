FROM python:3.8

ADD . /

RUN pip install pandas==1.3.3 numpy==1.21.2 scipy==1.7.1 scikit_learn==0.24.2 openpyxl timy

CMD ["python", "./main.py"]