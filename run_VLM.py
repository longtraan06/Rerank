import VLM
import load_top2


model, tokenizer, generation_config = VLM.load_model()
for index in range(6):
    query, captions_list, object_ids, query_id= load_top2.main()
    
    VLM.main(model, tokenizer, generation_config, object_ids, query)