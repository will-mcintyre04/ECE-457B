import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load the dataset
    df = pd.read_csv("A1Q4-decision-trees.csv")
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    split_configs = {
        "80/10/10": (0.80, 0.10, 0.10),
        "34/33/33": (0.34, 0.33, 0.33), 
        "25/25/50": (0.25, 0.25, 0.50)
    }

    results = []

    for split_name, (train_r, val_r, test_r) in split_configs.items():
        # split test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_r, random_state=123123
        )
        
        # split into train and validation
        val_ratio_adjusted = val_r / (train_r + val_r)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=123123
        )
        
        # Evaluate both criteria
        for crit in ['gini', 'entropy']:
            clf = DecisionTreeClassifier(max_depth=3, random_state=123123, criterion=crit)
            clf.fit(X_train, y_train)
            
            # Calculate accuracy scores
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            results.append({
                "Split": split_name,
                "Criterion": crit,
                "Train Acc": f"{train_acc:.4f}",
                "Val Acc": f"{val_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}"
            })

    # Print results in a readable format
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()