from errors import INVALID_ID, INVALID_DATA
from flask import Flask, jsonify, request
from helpers import Helper
from validate import Validate
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)
app = Flask('NewsAPI')
v = Validate()
h = Helper()
logging.info('App created!')


@app.route('/get_news/<int:user_id>', methods=['GET'])
def get_news(user_id):
    if v.valid_id(user_id):
        return '', 200
    return jsonify(INVALID_ID), 404


@app.route('/add_user', methods=['POST'])
def add_user():
    '''
    Принимает на вход данные вида: {"name": "John", "role": "booker"}
    Параметр "name": строка, передающая имя пользователя
    Параметр "role": строка, передающая роль пользователя
    Пример ответа: {"id": 1}
    Параметр "id": номер пользователя в системе
    '''
    logging.info('/add_user handled')
    data = request.get_json()
    logging.info('Data %s' % data)
    if v.valid_data(data):
        return jsonify(h.add_user(data)), 200
    return jsonify(INVALID_DATA), 400


if __name__ == '__main__':
    logging.info('Run app!')
    app.run()
