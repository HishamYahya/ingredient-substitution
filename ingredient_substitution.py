# Howdy! This is v1.0.0 :D


path_name = '/Users/cjxx/Downloads/summer/ingredient-substitution-model'


from substitution import Substitution
from gevent.pywsgi import WSGIServer
import json
from flask import Flask, request


def main():
    app = Flask(__name__)


    @app.route('/ingredient_substitution', methods=['POST'])
    def get_substitutions():
        sub = Substitution(path_name)
        request_data = request.get_json()
        ingredients = []
        for i in request_data['ingredients']:
            ingredients.append(i['ingredient_name'])
        return json.dumps(sub.get_substitutions(ingredients))


    @app.route('/', methods=['GET'])
    def index():
        return 'Welcome!'


    print("loading server")
    server = WSGIServer(('', 5000), app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.close()


if __name__ == '__main__':
    main()

