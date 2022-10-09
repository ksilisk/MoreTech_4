"""
Тут происходит вся валидацияа
"""

import logging
import backend.db.sqllib as sql


class Validate:
    '''
    Класс занимающийся валидацией данных от пользователя

    valid_data(self, data: dict) -> bool:
        функция, проверяющая на валидность входные данные Post-запроса
        data: dict - данные от пользователя

    valid_id(self, id: int) -> bool:
        функция, проверяющая на валидность id, который лежит в запросе
        id: int - номер пользователя в базе данныъ
    '''
    def __init__(self) -> None:
        pass

    def valid_data(self, data: dict) -> bool:
        logging.info("Validate valid_data func")
        if type(data['name']) == str and type(data['role']) == str \
                and data['role'] in ['booker', 'ceo'] and sql.check_name(data['name']):
            return True
        return False

    def valid_id(self, id: int) -> bool:
        return sql.check_id(id)
