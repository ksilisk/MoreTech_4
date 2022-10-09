import re

import numpy as np
import pandas as pd
import json

import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

import nltk
nltk.download("punkt")
nltk.download("stopwords")

import warnings
warnings.filterwarnings("ignore")
STOPWORDS = nltk.corpus.stopwords.words("russian")


class DataReader:

    """

    @decs
    Data reading class. Converts json to DataFrame if it is neccessary,
    or just reading from csv.

    @methods
    __init__(self, data_path) -> Initializes DataReader instance
    with exact data_path.

    """

    def __init__(
        self,
        data_path: str = ""
    ) -> None:

        """
        @desc DataReader contructor

        @param data_path: Raw data path

        @return: None (readed data as atribute)
        """

        self.data = None

        if len(data_path) > 0:
            assert data_path[-5:] == ".json" or data_path[-4:] == ".csv", \
                f"Data file extension is not supported {data_path}"

            if data_path[-5:] == ".json":
                with open(data_path) as json_data:
                    data = json.load(json_data)

                titles, texts, dates = [], [], []

                for data_key in data.keys():
                    for tag in data[data_key]:
                        for tag_key in tag.keys():
                            if tag_key == "title":
                                titles.append(tag[tag_key])
                            elif tag_key == "text":
                                texts.append(tag[tag_key])
                            elif tag_key == "date":
                                dates.append(tag[tag_key])

                data_dict = {
                    "title": titles,
                    "text": texts,
                    "date": dates
                }

                self.data = pd.DataFrame(data_dict)

            elif data_path[-4:] == ".csv":
                self.data = pd.read_csv(data_path)
                self.data = self.data.drop(columns=["Unnamed: 0"])
        else:
            self.data = pd.read_csv("./data/lenta-ru-news.csv")
            self.data.drop(columns=["url", "topic"], inplace=True)

        self.data["train"] = self.data["title"] + ". " + self.data["text"]


class TextVectorizer:

    """
    
    @desc
    Class for performing vectorization of text data
 
    @method __init__(self, data, stopwords) -> Initializes Vectorizer instance
    @method preprocess_data(self, source_sentence, remove_stopwords) -> Deletes
    dots, commas and etc.
    @methodsembed_bert_cls(self, text, model, tokenizer) -> Generates embeddings
    from text data
    @method fit(self) -> Fits the vectorizer

    """

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame,
        stopwords: set = None
    ) -> None:

        """
        @desc TextVectorizer contructor

        @param data: raw data path
        @param stopwords: set of russian stopwords

        @return: None
        """

        self.data = data
        self.stopwords = stopwords
    
    def preprocess_data(
        self,
        source_sentence: str,
        remove_stopwords: bool
    ) -> str:

        """
        @desc Deletes dots, commas and etc.

        @param source_sentence: Sentence to proced
        @param remove_stopwords: Flag to know if working with stopwords or not

        @return Processed text
        """

        source_sentence = re.sub(r'[^\w\s]', '', str(source_sentence))

        if remove_stopwords:
            tokens = nltk.word_tokenize(source_sentence, language="russian")
            tokens = [word for word in tokens if not word.lower() in self.stopwords]
            source_sentence = " ".join(tokens)

        source_sentence = source_sentence.lower().strip()

        return source_sentence

    def embed_bert_cls(
        self, 
        text: str, model: AutoModel,
        tokenizer: AutoTokenizer
    ) -> torch.nn.functional:

        """
        @desc Generates embedding from text data

        @param text: Source text to proced
        @param model: Model for performing tokenization
        @param tokenizer: Source tokenizer

        @return Embedded text
        """

        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{key: value.to(model.device) for key, value in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def fit(self) -> None:

        """
        @desc Fits tokenizer

        @return None
        """

        tokenized_data = [
            self.preprocess_data(
                source_sentence=sentence, remove_stopwords=True
            )
            for sentence in self.data["text"]
        ]

        self.data["embeded_text"] = [
            self.embed_bert_cls(text, model, tokenizer)
            for text in tokenized_data
        ]


class EstimateKMeans:

    """
    
    @desc Class for KMeans clustering estimation

    @method __init__(self, data, n_clusters) -> Initialize KMeans instance
    @method fit(self) -> Fitting KMeans clusterer
    
    """

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame,
        n_clusters: int = 3
    ) -> None:

        """
        @desc Initialize KMeans clusterer

        @param data: Data to cluster
        @param n_clusters: Number of cluster to distinguish

        @return None
        """

        self.n_clusters = n_clusters
        self.data = data
        self.clusters = None

    def fit(self) -> None:

        """
        @desc Fitting KMeans clusterer

        @return None (fitted clusterer)
        """

        embed_matrix = np.array(self.data["embeded_text"].tolist())

        k_means = KMeans(n_clusters=self.n_clusters, max_iter=100)
        k_means.fit(embed_matrix)

        self.clusters = k_means.labels_

        self.data['cluster'] = pd.Series(self.clusters)
        self.data['centroid'] = self.data['cluster'].apply(lambda x: k_means.cluster_centers_)


class TrendDetector:

    """
    
    @desc Class for performing trend detection

    @method __init__(self, data, clusters) -> Inintialize trend detector instance
    @method data_time_converter(self) -> Convert data to datetime format
    @method time_series_converter(self, data) -> Creates time series from data
    @method trend_detection_be_date(self, week_date) -> Performing trend detection by date

    """

    def __init__(
        self,
        data,
        clusters: list = []
    ) -> None:

        """
        @desc Initialize trend detector instance

        @param data: Embeded data
        @param clusters: List of distinguished clusters

        @return None
        """

        self.data = data
        self.clusters = clusters

    def data_time_converter(self) -> None:

        """
        @desc Conver data to datetime format

        @return None
        """

        dates = np.array(self.data.date)

        for i in range(len(dates)):
            dates[i] = dates[i][:10]
            dates[i] = dates[i].replace('.', '-')
            if dates[i][4] != '-':
                d, m, y = dates[i][:2], dates[i][3:5], dates[i][-4:]
                dates[i] = y + '-' + m + '-' + d

        self.data.date = dates

        date_converted = pd.to_datetime(pd.to_datetime(self.data.date).dt.date)

        self.data.date = date_converted

    def time_series_converter(self, data: pd.DataFrame) -> pd.Series:

        """
        @desc Creates time series from data

        @param data: Source dataset

        @return time series of type pd.Series
        """

        date_col = pd.to_datetime(pd.to_datetime(data.date).dt.date)
        time_series = date_col.value_counts()
        time_series = pd.DataFrame(time_series.values, index=time_series.index)

        time_series = time_series.sort_index()

        return time_series[0]

    def trend_detection_by_date(
        self,
        week_date: str
    ) -> pd.DataFrame:

        """
        @desc Performing trend detection by date

        @param week_data: Exact date to detect trend

        @return List of trended clusters (texts` themes)
        """

        self.data_time_converter()
        week_date = datetime.datetime.strptime(week_date, "%Y-%m-%d").date()

        week_ago = week_date - datetime.timedelta(days = 7)
        week_date, week_ago = pd.Timestamp(week_date), pd.Timestamp(week_ago)
        last_week = self.data[(self.data["date"] <= week_date) & (self.data["date"] >= week_ago)]

        clusters_list = np.unique(self.clusters)
        clusters_trend = pd.DataFrame(clusters_list, columns=['cluster'])
        clusters_trend['trending'] = 0

        max_cluster, max_publications = 0, 0

        for cluster in clusters_list:
            sample = last_week[last_week.cluster == cluster]
            time_series = self.time_series_converter(sample)

            if time_series[-1] > max_publications:
                max_cluster = cluster
                max_publications = time_series[-1]

            mean, std = float('nan'), float('nan')

            i = 7
            while pd.isna(mean) and i >= 1:
                rolling_mean = time_series.rolling(i).mean()
                rolling_std = time_series.rolling(i).std()
                mean = (time_series[-1] - rolling_mean[-1]) / (rolling_std[-1] + 1e-9)
                i -= 1
            
            i = 7
            while pd.isna(std) and i >= 1:
                rolling_std_mean = rolling_std.rolling(i).mean()
                rolling_std_std = rolling_std.rolling(i).std()
                std = (rolling_std[-1] - rolling_std_mean[-1]) / (rolling_std_std[-1] + 1e-9)
                i -= 1
            
            if pd.isna(std):
                std = mean - 1

            if mean > 0 and std > 0:
                clusters_trend['trending'][clusters_trend['cluster'] == cluster] = 1

        if all(clusters_trend.trending == 0):
            clusters_trend.trending[clusters_trend.cluster == max_cluster] = 1

        return clusters_trend
    


def main() -> None:
    reader = DataReader(data_path=data_path)
    data = reader.data

    vectorizer = TextVectorizer(data=data, stopwords=STOPWORDS)
    vectorizer.fit()
    vect_data = vectorizer.data

    clusterer = EstimateKMeans(data=vect_data, n_clusters=5)
    clusterer.fit()
    clusters = clusterer.clusters

    trend_detector = TrendDetector(data=vect_data, clusters=clusters)
    trends = trend_detector.trend_detection_by_date("2022-10-04")   
    print(trends)

if __name__ == "__main__":
    data_path = "./data/result_acc.json"
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    main()
