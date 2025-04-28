from .modules import VLM
import pandas as pd

class Reranker:
    def __init__(self):
        self.model, self.tokenizer, self.generation_config = VLM.load_model()
        self.reranked = []

    def rerank(self, Rerank_list):

        objects_path = [
            f"{self.private_dir}/objects/{Rerank_list[1]}/image.jpg",
            f"{self.private_dir}/objects/{Rerank_list[2]}/image.jpg"
        ]
        reranked_list = VLM.main(
            self.model,
            self.tokenizer,
            self.generation_config,
            objects_path,
            Rerank_list,
            self.query,
            Rerank_list[0]
        )
        return reranked_list

    def __call__(self, private_dir, Rerank_list, 
                 #query
                 ):
        self.private_dir = private_dir

        query_path = f"{self.private_dir}/scenes/{Rerank_list[0]}/query.txt"
        with open(query_path, "r", encoding="utf-8") as file:
            query = file.read()
        self.query = query
        reranked_list = self.rerank(Rerank_list)
        return reranked_list



