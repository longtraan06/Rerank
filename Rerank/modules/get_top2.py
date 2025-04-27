import csv
import json
from collections import defaultdict

def csv_to_json(csv_file_path, json_file_output_path):
    # Tạo dictionary để lưu trữ dữ liệu theo query
    query_data = defaultdict(list)
    
    # Đọc file CSV
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Lấy tên các cột (giả sử cột đầu tiên là 'query' và 10 cột tiếp theo là output1 đến output10)
        fieldnames = csv_reader.fieldnames
        output_columns = fieldnames[1:11]  # Lấy 10 cột output đầu tiên
        
        for row in csv_reader:
            query_name = row[fieldnames[0]]  # Lấy tên query từ cột đầu tiên
            
            # Lấy 2 output đầu tiên
            top_2_outputs = [row[col] for col in output_columns[:2]]
            
            # Thêm vào dictionary
            query_data[query_name] = top_2_outputs
    
    # Ghi dữ liệu ra file JSON
    with open(json_file_output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(query_data, json_file, ensure_ascii=False, indent=4)
    
   # print(f"Đã tạo file JSON thành công tại: {json_file_output_path}")

def main(csv_file_path, json_file_output_path):

    csv_to_json(csv_file_path, json_file_output_path)