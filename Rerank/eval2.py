import csv
import io
import pandas as pd
# Load groundtruth CSV once
def load_groundtruth(path='Evaluate.csv'):
    groundtruth_list = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            groundtruth_list.append(row)
    return groundtruth_list

# Evaluation logic
def evaluate_file(test_csv_path, groundtruth_list):
    with open(test_csv_path, mode='r') as file:
        content = file.read()
        reader = csv.reader(io.StringIO(content))
        test_list = [row for row in reader]

    # Scoring
    score = 0
    false_list = []
    special_ids = {
        '1b6f6bfa-a78f-4df4-9bd6-526fe11a6d30',
        '48eb9d73-6105-47cd-a5c3-e147973cac19',
        '69d0546d-a03b-4c82-a516-3bb1b76c0c6b',
        'df57b527-4f5b-4be2-b443-ae85d8e23ef8',
        'd8ebcd50-475b-43c0-9c5e-d5d84bb9d31a'
    }

    for scenetruth in groundtruth_list:
        for scenetest in test_list:
            if scenetruth[0] == scenetest[0]:
                target_ids = [scenetruth[1]]
                if scenetest[0] in special_ids and len(scenetruth) > 2:
                    target_ids.append(scenetruth[2])

                found = False
                for idx in range(1, min(len(scenetest), 11)):  # Only top-10
                    if scenetest[idx] in target_ids:
                        score += 1 / idx
                        if idx >= 2:
                            scenetest.append('Ranked at ' + str(idx))
                            false_list.append(scenetest)
                        found = True
                        break
                if not found:
                    scenetest.append('Not Found')
                    false_list.append(scenetest)
                    score += 0

    total_score = (score / 50)  # Normalize over 50 items
    return {
        "score": round(total_score, 4),
        "false_scenes": false_list
    }

# Run the evaluator
if __name__ == "__main__":
    groundtruth = load_groundtruth('Evaluate.csv')
    result = evaluate_file('/root/Rerank/important_code/reranked_file.csv', groundtruth)
    print("Score:", result["score"])
    print("\nFalse Scenes:")
    df = pd.DataFrame(result['false_scenes'])
    df.to_csv("false_room.csv", index = False)
