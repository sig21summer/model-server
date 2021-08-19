from flask import Flask, request
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd

import model

app = Flask(__name__)


@app.route('/')
def root():
    return 'hello'


@app.route("/model")
def root2():
    json = request.get_json()
    text = json['text']
    if (json.get('count')):
        k = json['count']
    else:
        k = 5

    res = model.predictNext(text, k)
    res_json = {}
    res_json["data"] = res
    return res_json


if __name__ == "__main__":
    app.run()
