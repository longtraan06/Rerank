import load_top2
import get_top2
import get_query
import VLM
root_dir = "/home/liex/Desktop/rerank" # replace with your root folder dir
csv_path = "/home/liex/Desktop/rerank/MealsRetrieval0.9283.csv" # replace with your csv path
private_dir = "/home/liex/Desktop/rerank/private"  # replace with your private folder path

model, tokenizer, generation_config = VLM.load_model()

output_top2_path = root_dir + "/top2.json"
get_top2.main(csv_path, output_top2_path)

text_query_path =  root_dir + "/query_texts.json"
get_query.main(output_top2_path, private_dir + "/scenes", text_query_path)

# rerank in here
for index in range(50):
    #get text query, object_ids_path (object name), query id( query name )
    query, object_ids, query_id = load_top2.main(index, text_query_path, output_top2_path)
    #create query image path
    query_image_path = private_dir + "/scenes/" + query_id + "/masked.png"
    #create object image path of top 2 
    objects_path = []
    for i in range(2):
        objects_path[i] = private_dir + "/objects/" + objects_path[i] + "/image.jpg"
    """
        tổng tất cả các biến có thể sử dụng là :
            query : text desciption
            query_image_path : ảnh panoramic
            object_ids_path : list 2 path dẫn đến ảnh của top 2 object
    """
    # reraking
    VLM.main(model, tokenizer, generation_config, objects_path, object_ids, query)



    # # print test
    # print("text query: ", query)
    # print("panoramic image path: ", query_image_path)
    # print("top 2 object paths: ")
    # for path in object_ids_path:
    #     print(path)
    # break
    

