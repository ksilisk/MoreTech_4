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
    def parse_tg(self, name: str) -> list:
        result = []
        for filename in self.sites[name]:
            with open('tg_channel/' + filename, 'r') as f:
                page = f.read()
            f.close()
            soup = BeautifulSoup(page, 'html.parser')
            data = soup.find_all('div', 'message default clearfix')
            for news in data:
                if len(news.find('div', 'text').find_all('a')) == 1 and news.find('div', 'text').find('strong'):
                    if len((news.find('div', 'text').find('a')['href']).split('/')) == 5 \
                            and (news.find('div', 'text').find('a')['href']).split('/')[2] == 'telegra.ph':
                        title = news.find('div', 'text').find('strong').text.replace("\xa0", " ").replace("\u200b", "")
                        date = news.find('div', 'pull_right date details')['title']
                        date = date[:19]
                        text = news.find('div', 'text').text.replace("\xa0", " ").replace("\u200b", "")
                        result.append({'title': title, 'date': date, 'text': text})
        return result

    def kba(self) -> list:
        stop_word = ['REAL CAPITAL', 'ПОДПИШИСЬ', 'ГАРАНТИРУЕМ', 'подпишись', 'гарантируем', 'RAL CAPITAL']
        result = []
        for filename in self.sites['kba']:
            with open('tg_channel/' + filename, 'r') as f:
                page = f.read()
            f.close()
            soup = BeautifulSoup(page, 'html.parser')
            data = soup.find_all('div', 'message default clearfix')
            for news in data:
                if news.find('div', 'text') and news.find('div', 'text').find('a') is None:
                    text = news.find('div', 'text').text.replace('\u200d', "")
                    flag = True
                    for word in stop_word:
                        if word in text:
                            flag = False
                    if flag:
                        title = text.split('📌')[0]
                        date = news.find('div', 'pull_right date details')['title']
                        result.append({'title': title, 'date': date, 'text': text})
        return result



def main():
    parser = Parser()
    a = parser.kba()
    for i in a:
        print(i)


if __name__ == '__main__':
    main()
