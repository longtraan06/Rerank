import json
import os

def extract_query_texts_from_top2(top2_json_path, object_folder, output_json_path):
    # Bước 1: Đọc file top4 JSON
    with open(top2_json_path, 'r', encoding='utf-8') as f:
        top2_data = json.load(f)
    
    # Bước 2: Lấy query text cho các query ID có trong top5
    query_texts = {}
    
    for query_id in top2_data.keys():
        query_file_path = os.path.join(object_folder, query_id, "query.txt")
        
        if os.path.exists(query_file_path):
            with open(query_file_path, 'r', encoding='utf-8') as f:
                query_texts[query_id] = f.read().strip()
        else:
            print(f"Warning: Không tìm thấy query.txt cho {query_id}")
    
    # Bước 3: Ghi ra file JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(query_texts, f, ensure_ascii=False, indent=4)
    
   # print(f"Đã lưu query texts vào {output_json_path}")

# Sử dụng
def main(top2_json_path, scenes_folder, output_json_path):

    extract_query_texts_from_top2(top2_json_path, scenes_folder, output_json_path)