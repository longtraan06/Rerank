from .modules import get_query, get_top2, load_top2, VLM
import pandas as pd

class Reranker:
    def __init__(self):
        self.model, self.tokenizer, self.generation_config = VLM.load_model()
        self.reranked = []

    def prepare_top2_and_queries(self):
        get_top2.main(self.csv_path, self.output_top2_path)
        get_query.main(self.output_top2_path, f"{self.private_dir}/scenes", self.text_query_path)

    def rerank(self, num_queries=50):
        for index in range(num_queries):
            query, object_ids, query_id = load_top2.main(index, self.text_query_path, self.output_top2_path)
            
            query_image_path = f"{self.private_dir}/scenes/{query_id}/masked.png"
            objects_path = [
                f"{self.private_dir}/objects/{object_ids[0]}/image.jpg",
                f"{self.private_dir}/objects/{object_ids[1]}/image.jpg"
            ]
            
            rerank_part = VLM.main(
                self.model,
                self.tokenizer,
                self.generation_config,
                objects_path,
                object_ids,
                query,
                query_id,
                query_image_path
            )
            self.reranked.append(rerank_part)

    def save_reranked_results(self, output_csv='reranked_file.csv'):
        df = pd.read_csv(self.csv_path, header=None)
        rerank_dict = {item['query_id']: item['objects'] for item in self.reranked}
        
        def update_row(row):
            query_id = row[0]
            if query_id in rerank_dict:
                new_top2 = rerank_dict[query_id]
                remaining_items = row[3:11].tolist()
                updated_row = [query_id] + new_top2 + remaining_items
                if len(updated_row) < 11:
                    updated_row += [''] * (11 - len(updated_row))
                return pd.Series(updated_row)
            return row
        
        updated_df = df.apply(update_row, axis=1)
        updated_df.to_csv(output_csv, index=False, header=False)

    def run(self, num_queries=50, output_csv='reranked_file.csv'):
        self.prepare_top2_and_queries()
        self.rerank(num_queries)
        self.save_reranked_results(output_csv)

    def __call__(self, csv_path, private_dir):
        self.csv_path = csv_path
        self.private_dir = private_dir
        self.output_top2_path = "./top2.json"
        self.text_query_path = "./query_texts.json"
        self.reranked = []
        self.run(num_queries=50, output_csv="reranked_file.csv")
        