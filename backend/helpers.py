import db.sqllib as sql
import logging
import numpy as np
import pandas as pd


class Helper:
    def __init__(self) -> None:
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
