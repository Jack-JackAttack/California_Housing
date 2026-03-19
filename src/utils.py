import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score



def print_cv_results(to_print_dict):
    for name, model_scores in to_print_dict.items():
            print(name)
            print("f1 för varje fold:", model_scores.round(3))
            print("Medelvärdet för alla 5 folds:",model_scores.mean().round(3))
            print("Standardavvikelse:", model_scores.std().round(3),"\n")



def metric_table(y_true, y_pred, model_name="Name of model"):
    
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
        return pd.DataFrame([{
            "Model": model_name,
            "Precision" :round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1,3)
        }])