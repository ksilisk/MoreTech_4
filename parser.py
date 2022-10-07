'''
Тут класс, в котором реализованы функции парсинга новостных сайтов
'''
from selenium import webdriver
class Parser:

    def __init__(self) -> None:
        self.sites = {'1': 'https://www.vedomosti.ru/'}
        self.driver = webdriver.PhantomJS()

    # парсинг сайта ведомостей
    def vedomosti(count: int) -> dict:
        pass