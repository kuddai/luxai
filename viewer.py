#!/usr/bin/env python3

import glob
import json
import os
import werkzeug.security
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__,
            static_url_path='',
            static_folder='viewer/dist')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# uncomment for debug
app.config['DEBUG'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

root = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def static_files():
    return send_from_directory('viewer/dist', 'index.html')

# A route to return all of the available entries in our catalog.
@app.route('/api/v1/replay', methods=['GET'])
@cross_origin()
def replays():
    replays_dir = root + '/replays'
    replays_glob = replays_dir + '/**/*.json'
    from os.path import relpath
    replays_paths = [relpath(p, replays_dir) for p in glob.iglob(replays_glob, recursive=True)]
    return jsonify(replays_paths)

@app.route('/api/v1/replay/<path:path>', methods=['GET'])
@cross_origin()
def replay(path):
    print('kuddai path individual', path)
    abs_path = werkzeug.security.safe_join(root, 'replays/' + path)
    print('kuddai abs path', abs_path)
    with open(abs_path) as f:
        return jsonify(json.load(f))

# Default Flask port is 5000
# 0.0.0.0 ~= localhost:5000
app.run(host='0.0.0.0')
