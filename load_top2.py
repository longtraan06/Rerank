import json

def load_query_top2_data(json_file_path):
    """Load JSON file and return as dictionary {query: top4_list}"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_caption_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    return captions

def load_query_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        querys = json.load(f)
    return querys
# Sử dụng
def main(index, query_json, json_file_path):

    query_top2_data = load_query_top2_data(json_file_path)
    text_query_top2 = load_query_json(query_json)
    # Lấy danh sách tất cả các query
    all_queries = list(query_top2_data.keys())

    # Lấy top2 của query đầu tiên
    first_query = all_queries[index]
    first_top2 = query_top2_data[first_query]
    return text_query_top2[first_query], first_top2, first_query

