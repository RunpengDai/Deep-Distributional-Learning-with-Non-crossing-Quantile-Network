FROM python:3.6.13

WORKDIR /usr/src/app

COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install atari-py
RUN python -m atari_py.import_roms /usr/src/app/Atari
RUN pip install  -r requirements.txt --verbose

