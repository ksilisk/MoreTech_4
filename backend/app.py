from backend.errors import INVALID_ID, INVALID_DATA
from flask import Flask, jsonify, request
from backend.helpers import Helper
from backend.validate import Validate
from ML.model import get_trends as trends
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)
app = Flask('NewsAPI')
app.config['JSON_AS_ASCII'] = False
v = Validate()
h = Helper()
logging.info('App created!')


@app.route('/get_news/<int:user_id>', methods=['GET'])
def get_news(user_id):
    '''
    Функция, выдающая новость пользователю
    Принимает на вход:
    user_id: int - номер пользователя в системе
    Пример ответа: [{'title': 'some_title', 'text': 'some_text'}]
    "title": str - заголовок новости
    "text": str - текст новости
    '''
    logging.info('/get_news handled')
    if v.valid_id(user_id):
        detector = TrendDetector()
    return jsonify(INVALID_ID), 404


@app.route('/add_user', methods=['POST'])
def add_user():
    '''
    Функция, добавляющая нового пользователя в базу
    Принимает на вход данные вида: {"name": "John", "role": "booker"}
    "name": str, - передает имя пользователя
    "role": str, - передает роль пользователя
    Пример ответа: {"id": 1}
    "id": int - номер пользователя в системе
    '''
    logging.info('/add_user handled')
    data = request.get_json()
    logging.info('Data %s' % data)
    if v.valid_data(data):
        return jsonify(h.add_user(data)), 200
    return jsonify(INVALID_DATA), 400


@app.route('/get_trends/<int:user_id>', methods=['GET'])
def get_trends(user_id):
    if v.valid_id(user_id):
        date = '2022-10-04'
        return jsonify(h.trends_to_json(user_id, trends(date))), 200
    return jsonify(INVALID_ID), 404


if __name__ == '__main__':
    logging.info('Run app!')
    app.run()
