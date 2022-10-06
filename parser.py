'''
Тут класс, в котором реализованы функции парсинга новостных сайтов
'''

class Parser:

    def __init__(self) -> None:
        self.sites = {'1': 'https://www.vedomosti.ru/'}

    # парсинг сайта ведомостей
    def vedomosti(count: int) -> dict:
        pass