from flask import Flask, request
from parser import CheckboxParser
import tempfile
from PIL import Image
import os

app = Flask(__name__)


@app.route("/", methods=['POST'])
def index():
    if request.method == 'POST':
        stream = request.files.get('file', '')
        type = request.form['type']
        fh, input = tempfile.mkstemp()
        with open(input+".pdf", "wb") as f:
            f.write(stream.read())
            f.close()
        obj = CheckboxParser(input+".pdf", "*")
        result = obj.checkbox_parser()
        return result
app.run()


