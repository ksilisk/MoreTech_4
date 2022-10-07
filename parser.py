'''
Тут класс, в котором реализованы функции парсинга новостных сайтов
'''
from bs4 import BeautifulSoup
import requests
import json
import feedparser
import time

class Parser:

    def __init__(self) -> None:
        with open('sites.json', 'r') as f:
            self.sites = json.loads(f.read())

    #  парсинг сайта ведомостей
    def buh_ru(self, count: int) -> list:
        last_news_id = int(feedparser.parse(self.sites['buh_ru'])['entries'][0]['links'][0]['href'].split('/')[5])
        if count >= last_news_id:
            return []
        result = []
        for i in range(last_news_id - count, last_news_id):
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


def main():
    parser = Parser()
    a = parser.buh_ru(10)
    for i in a:
        print(i['date'] + '\n', i['text'])


if __name__ == '__main__':
    main()
