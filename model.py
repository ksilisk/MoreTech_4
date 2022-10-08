import re

import numpy as np
import pandas as pd
import json

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

import nltk
nltk.download("punkt")
nltk.download("stopwords")
STOPWORDS = nltk.corpus.stopwords.words("russian")


class DataReader:
    def __init__(
        self,
        data_path: str = ""
    ) -> None:
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

                print(len(titles), len(texts), len(dates))

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
    def __init__(
        self,
        data: pd.DataFrame,
        stopwords: set
    ) -> None:
        self.data = data
        self.stopwords = stopwords
    
    def preprocess_data(
        self,
        source_sentence: str,
        remove_stopwords: bool,
        stopwords: set
    ) -> str:

        source_sentence = re.sub(r'[^\w\s]', '', str(source_sentence))

        if remove_stopwords:
            tokens = nltk.word_tokenize(source_sentence, language="russian")
            tokens = [word for word in tokens if not word.lower() in stopwords]
            source_sentence = " ".join(tokens)

        source_sentence = source_sentence.lower().strip()

        return source_sentence

    def embed_bert_cls(
        self, 
        text, model,
        tokenizer
    ) -> torch.nn.functional:
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{key: value.to(model.device) for key, value in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def fit(self) -> None:
        tokenized_data = [
            self.preprocess_data(
                source_sentence=sentence, remove_stopwords=True, stopwords=self.stopwords
            )
            for sentence in self.data["text"]
        ]

        self.data["embeded_text"] = [
            self.embed_bert_cls(text, model, tokenizer)
            for text in tokenized_data
        ]


class EstimateKMeans:
    def __init__(
        self,
        data = pd.DataFrame,
        n_clusters: int = 3
    ) -> None:
        self.n_clusters = n_clusters
        self.data = data
        self.clusters = None

    def fit(self) -> None:
        embed_matrix = np.array(self.data["embeded_text"].tolist())

        k_means = KMeans(n_clusters=self.n_clusters, max_iter=100)
        k_means.fit(embed_matrix)

        self.clusters = k_means.labels_

        self.data['cluster'] = pd.Series(self.clusters)
        self.data['centroid'] = self.data['cluster'].apply(lambda x: k_means.cluster_centers_)


def main() -> None:
    reader = DataReader(data_path=data_path)
    data = reader.data

    vectorizer = TextVectorizer(data=data, stopwords=STOPWORDS)
    vectorizer.fit()
    vect_data = vectorizer.data

    clusterer = EstimateKMeans(data=vect_data, n_clusters=5)
    clusterer.fit()
    clusters = clusterer.clusters    

if __name__ == "__main__":
    data_path = "./data/result_gendir.json"
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    main()
