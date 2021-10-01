#!/usr/bin/env python3

import glob
import json
import os
import argparse
import werkzeug.security
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__,
            static_url_path='',
            static_folder='Lux-Viewer-2021/dist')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# uncomment for debug
app.config['DEBUG'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

root = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPLAYS_DIR = os.path.join(root, 'replays')
replays_dir = DEFAULT_REPLAYS_DIR

@app.route('/')
def static_files():
    return send_from_directory('Lux-Viewer-2021/dist', 'index.html')

# A route to return all of the available entries in our catalog.
@app.route('/api/v1/replay', methods=['GET'])
@cross_origin()
def replays():
    replays_glob = replays_dir + '/**/*.json'
    from os.path import relpath
    print('kuddai replays_glob', replays_glob)
    replays_paths = [relpath(p, replays_dir) for p in glob.iglob(replays_glob, recursive=True)]
    return jsonify(replays_paths)

@app.route('/api/v1/replay/<path:path>', methods=['GET'])
@cross_origin()
def replay(path):
    abs_path = werkzeug.security.safe_join(replays_dir,  path)
    with open(abs_path) as f:
        return jsonify(json.load(f))


def main():
    global replays_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default=5000)
    parser.add_argument(
        '--replays', type=str, default=DEFAULT_REPLAYS_DIR,
        help='directory with replays recordings')
    args = parser.parse_args()
    replays_dir = args.replays
    print('using replays dir', replays_dir)
    # Default Flask port is 5000
    # 0.0.0.0 ~= localhost:5000
    app.run(host='0.0.0.0', port=args.port)

main()
