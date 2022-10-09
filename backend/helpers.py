from ML.model import top_k_news
from ML.model import TrendDetector
from ML.model import get_trends as trends
import backend.db.sqllib as sql
import logging
import numpy as np
import pandas as pd


class Helper:
    '''
    Класс проводящий вспомогательные операции с данными

    add_user(self, data: dict) -> dict:
        добавляет пользователя в базу и возвращает его id

    trends_to_json(self, user_id: int, trends: pd.DataFrame) -> dict:
        конвертирует тренды в json формат для дальнейшей отправки
    '''
    def __init__(self) -> None:

        # бухгалтерски тренды
        self.acc_clusters_dict = {
            0: 'Налоговое законодательство',
            1: 'Налоги и отчетность',
            2: 'Финансы',
            3: 'Торговля',
            4: 'Трудовое законодательство',
            5: 'Документооборот',
            6: 'Бизнес России и Мира',
            7: 'Программное обеспечение для бухгалтеров',
            8: 'Макроэкономика',
            9: 'OTHER'
        }

        # тренды генерального директора
        self.ceo_clusters_dict = {
            0: 'Финансы',
            1: 'Digital',
            2: 'Высокие технологии',
            3: 'Макроэкономика',
            4: 'Советы бизнесменам'
        }

    def add_user(self, data: dict) -> dict:
        logging.info("Helpers add_user func")
        return {"id": sql.add_user(data['name'], data['role'])}

    def trends_to_json(self, user_id: int, trends: pd.DataFrame) -> dict:
        role = sql.get_role(user_id)
        result = np.array(trends.index[trends.trending == 1])
        dict_data = {'trends': []}
        if role == 'booker':
            for i in result:
                dict_data['trends'].append(self.acc_clusters_dict[i])
        elif role == 'ceo':
            for i in result:
                dict_data['trends'].append(self.ceo_clusters_dict[i])
        return dict_data

    def news_to_json(self, user_id: int, date: str) -> dict:
        role = sql.get_role(user_id)
        trend_mark = trends(date)
        if role == 'booker':
            df = pd.read_csv("ML/data/dataframe_acc.csv")
            emb_acc = pd.read_csv("ML/data/embeddings_acc.csv").iloc[:, 1:]
            embed_keywords = np.array(pd.read_csv("ML/data/embed_acc_keywords.csv").iloc[:, 1])
            emb_array = np.array(emb_acc)
            df['date'] = pd.to_datetime(pd.to_datetime(df['date']).dt.date)
            df['rubert_tiny'] = np.split(emb_array, len(emb_array), axis=0)
        elif role == 'ceo':
            df = pd.read_csv("ML/data/df_ceo.csv")
            emb_ceo = pd.read_csv("ML/data/embeddings_ceo.csv").iloc[:, 1:]
            embed_keywords = np.array(pd.read_csv("ML/data/embed_ceo_keywords.csv").iloc[:, 1])
            emb_array = np.array(emb_ceo)
            df['date'] = pd.to_datetime(pd.to_datetime(df['date']).dt.date)
            df['rubert_tiny'] = np.split(emb_array, len(emb_array), axis=0)
        return top_k_news(embed_keywords, df, date, trend_mark)
