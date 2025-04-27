import load_top2
import get_top2
import get_query
import ecec
root_dir = "/root/Rerank" # replace with your root folder dir
csv_path = "/root/Rerank/MealsRetrieval0.9283.csv" # replace with your csv path
private_dir = "/root/Rerank/private"  # replace with your private folder path

# model, tokenizer, generation_config = ecec.load_model()

output_top2_path = root_dir + "/top2.json"
get_top2.main(csv_path, output_top2_path)

text_query_path =  root_dir + "/query_texts.json"
get_query.main(output_top2_path, private_dir + "/scenes", text_query_path)
reranked = []

# rerank in here
for index in range(50):
    #get text query, object_ids_path (object name), query id( query name )
    query, object_ids, query_id = load_top2.main(index, text_query_path, output_top2_path)
    #create query image path
    query_image_path = private_dir + "/scenes/" + query_id + "/masked.png"
    #create object image path of top 2 
    print(object_ids)
    print(query)
    objects_path = []
    objects_path.append(query_image_path)
    for i in range(2):
        objects_path. append(private_dir + "/objects/" + object_ids[i] + "/image.jpg")
    print(objects_path)
    """
        tổng tất cả các biến có thể sử dụng là :
            query : text desciption
            query_image_path : ảnh panoramic
            object_ids_path : list 2 path dẫn đến ảnh của top 2 object
    """
    # reraking
    

    rerank_part = ecec.main(objects_path, object_ids, query, query_id)
    reranked.append(rerank_part)

import pandas as pd

# 1. Đọc file CSV gốc
df = pd.read_csv('/root/Rerank/MealsRetrieval0.9283.csv', header=None)

# 2. Tạo dictionary từ kết quả rerank để tra cứu nhanh
rerank_dict = {item['query_id']: item['objects'] for item in reranked}

# 3. Hàm cập nhật từng dòng
def update_row(row):
    query_id = row[0]
    if query_id in rerank_dict:
        # Lấy top 2 mới từ kết quả rerank
        new_top2 = rerank_dict[query_id]
        
        # Giữ nguyên các item từ vị trí thứ 3 trở đi
        remaining_items = row[3:11].tolist()  # Các cột từ 3 đến 10 (index 2:10)
        
        # Tạo hàng mới: query_id + new_top2 + các item còn lại
        updated_row = [query_id] + new_top2 + remaining_items
        
        # Đảm bảo đủ 11 cột (query_id + 10 items)
        if len(updated_row) < 11:
            updated_row += [''] * (11 - len(updated_row))
        
        return pd.Series(updated_row)
    return row

# 4. Áp dụng cập nhật cho toàn bộ DataFrame
updated_df = df.apply(update_row, axis=1)

# 5. Ghi ra file mới
updated_df.to_csv('r1ranked_file.csv', index=False, header=False)

    

