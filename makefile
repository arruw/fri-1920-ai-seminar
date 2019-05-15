SHELL := /bin/bash

install:
	python3 -m venv .env/
	source .env/bin/activate;
	pip install -r requirements.txt;

train-taxi:
	mkdir -p .tmp/taxi; 
	python taxi/train.py

eval-taxi:
	python taxi/eval.py

demo-taxi:
	python taxi/demo.py