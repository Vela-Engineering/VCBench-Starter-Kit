import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, fbeta_score
import os
import json
import ast

def safe_eval(prediction_json):
    # Remove '''json at the front and ''' at the back
    if pd.isna(prediction_json) or prediction_json == None:
        return None
    if prediction_json.startswith("```json\n"):
        prediction_json = prediction_json[8:]  # Remove '''json
    if prediction_json.endswith("\n```"):
        prediction_json = prediction_json[:-4]  # Remove '''
    try:
        return ast.literal_eval(prediction_json)
    except:
        return json.loads(prediction_json) 

        
# Read the batch 1 predictions file
for file in sorted(os.listdir("vanilla_llm_testing_results")):
    df = pd.read_csv(os.path.join("vanilla_llm_testing_results", file))
    
    for index, row in df.iterrows():
        prediction_json = safe_eval(row['prediction'])
        if prediction_json == {} or pd.isna(prediction_json) or prediction_json == None:
            print(row['founder_uuid'])
            print(row['prediction'])
            print("empty response")
            prediction = "no"
        else:
            prediction = prediction_json['prediction']

        
    
        if prediction.lower() == 'yes':
            prediction_numeric = 1
        elif prediction.lower() == 'no':
            prediction_numeric = 0
        else:
            raise Exception(f"Invalid prediction: {prediction}")

        df.at[index, 'prediction_numeric'] = prediction_numeric

    # Calculate metrics using numeric predictions
    precision = precision_score(df['success'], df['prediction_numeric'])
    accuracy = accuracy_score(df['success'], df['prediction_numeric'])
    recall = recall_score(df['success'], df['prediction_numeric'])
    f_half_score = fbeta_score(df['success'], df['prediction_numeric'], beta=0.5)

    print(f"Metrics for {file}:")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F_0.5 score: {f_half_score:.4f}")

    # Print additional information
    print(f"\nTotal 2samples analyzed: {len(df)}")

    # Create a confusion matrix
    confusion_matrix = pd.crosstab(df['success'], df['prediction_numeric'], rownames=['Actual'], colnames=['Predicted'])
    print("\nConfusion Matrix:")
    print(confusion_matrix)