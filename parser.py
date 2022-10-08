'''
Тут класс, в котором реализованы функции парсинга новостных сайтов
'''
from telethon import TelegramClient, sync
from bs4 import BeautifulSoup
import requests
import json
import feedparser
import time


class Parser:

    def __init__(self) -> None:
        with open('sites.json', 'r') as f:
            self.sites = json.loads(f.read())
        f.close()
        with open('api.json', 'r') as f:
            self.api_data = json.loads(f.read())
        f.close()

    #  парсинг сайта buh.ru
    def buh_ru(self, start: int, count: int) -> list:
        last_news_id = int(feedparser.parse(self.sites['buh_ru'])['entries'][0]['links'][0]['href'].split('/')[5])
        if count >= last_news_id or count == 0:
            return []
        result = []
        for i in range(last_news_id - start - count, last_news_id - start):
            page = requests.get('https://buh.ru/news/uchet_nalogi/' + str(i))
            if page.status_code == 200:
                soup = BeautifulSoup(page.text, 'html.parser')
                title = soup.find('h1', class_='margin_line-height phead specdiv').text
                date = soup.find('span', class_='grayd').text
                text_data = soup.find('div', class_='tip-news').find_all('p')
                text = ''
                for j in range(len(text_data) - 1):
                    text += text_data[j].text + '\n'
                result.append({'title': title, 'date': date, 'text': text})
                time.sleep(3)
        return result

    #  парсинг сайта klerk.ru
    def klerk_ru(self, start: int, count: int) -> list:
        last_news_id = int(feedparser.parse(self.sites['klerk_ru'])['entries'][0]['links'][0]['href'].split('/')[5])
        if count >= last_news_id or count == 0:
            return []
        result = []
        for i in range(last_news_id - start - count, last_news_id - start):
            page = requests.get('https://www.klerk.ru/buh/news/' + str(i))
            if page.status_code == 200:
                soup = BeautifulSoup(page.text, 'html.parser')
                title = soup.find('header', 'article__header').find('h1').text
                date = soup.find('span', 'status__block').find('core-date-format')['date']
                text_data = soup.find('div', 'article__content').find_all()
                text = ''
                for j in text_data:
                    if j.name == 'p':
                        text += j.text + '\n'
                text = text.replace("\t", "").replace("\r", "").replace("\xa0", " ")
                result.append({'title': title, 'date': date, 'text': text})
        return result

    #  парсинг телеграмм каналов
    def parse_telegram_channel(self, count: int) -> list:
        client = TelegramClient('Parser', self.api_data['api_id'], self.api_data['api_hash'])


def main():
    parser = Parser()
    a = parser.klerk_ru(0, 10)
    for i in a:
        print(i)


if __name__ == '__main__':
    main()
