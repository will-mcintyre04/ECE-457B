import pandas as pd
import numpy as np

def calculate_gini(labels):
    if len(labels) == 0:
        return 0.0
    p1 = sum(labels) / len(labels)
    p0 = 1 - p1
    return 1 - (p0**2 + p1**2)

def calculate_entropy(labels):
    if len(labels) == 0:
        return 0.0
    p1 = sum(labels) / len(labels)
    p0 = 1 - p1
    
    entropy = 0.0
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    return entropy

def evaluate_split(df, split_condition, metric_func):
    # Evaluates the weighted impurity of a split based on a boolean condition
    left_node = df[split_condition]['Survived']
    right_node = df[~split_condition]['Survived']
    
    N = len(df)
    left_weight = len(left_node) / N
    right_weight = len(right_node) / N
    
    weighted_score = (left_weight * metric_func(left_node)) + (right_weight * metric_func(right_node))
    return weighted_score

def main():
    # Load the dataset
    df = pd.read_csv("A1Q4-decision-trees.csv")

    # compare gini
    gender_gini = evaluate_split(df, df['Gender'] == 0, calculate_gini)
    age_gini = evaluate_split(df, df['Age'] <= 65, calculate_gini)

    print(f"Weighted Gini (Gender): {gender_gini:.4f}")
    print(f"Weighted Gini (Age <= 65): {age_gini:.4f}")

    # compare shannon
    gender_entropy = evaluate_split(df, df['Gender'] == 0, calculate_entropy)
    age_entropy = evaluate_split(df, df['Age'] <= 65, calculate_entropy)

    print(f"\nWeighted Entropy (Gender): {gender_entropy:.4f}")
    print(f"Weighted Entropy (Age <= 65): {age_entropy:.4f}")

if __name__ == "__main__":
    main()