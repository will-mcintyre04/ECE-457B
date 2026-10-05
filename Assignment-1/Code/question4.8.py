import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("A1Q4-decision-trees.csv")
X = df.drop(columns=['Survived'])
y = df['Survived']

# Define splits
split_configs = {
    "80/10/10": (0.80, 0.10, 0.10),
    "34/33/33": (0.34, 0.33, 0.33), 
    "25/25/50": (0.25, 0.25, 0.50)
}

depths = range(1, 16)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Accuracy vs. Max Depth Across Splits", fontsize=16)

# Iterate through splits and calculate accuracies
for i, (split_name, (train_r, val_r, test_r)) in enumerate(split_configs.items()):
    # Split the data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_r, random_state=123123
    )
    val_ratio_adjusted = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=123123
    )
    
    train_accs, val_accs, test_accs = [], [], []
    
    # Test different depths
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=123123, criterion='gini')
        clf.fit(X_train, y_train)
        
        train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        val_accs.append(accuracy_score(y_val, clf.predict(X_val)))
        test_accs.append(accuracy_score(y_test, clf.predict(X_test)))
        
    # Plotting
    ax = axes[i]
    ax.plot(depths, train_accs, label='Train', marker='o', linestyle='-')
    ax.plot(depths, val_accs, label='Val', marker='o', linestyle='-')
    ax.plot(depths, test_accs, label='Test', marker='o', linestyle='-')
    
    ax.set_title(f"Split: {split_name}")
    ax.set_xlabel("Max Depth")
    if i == 0:
        ax.set_ylabel("Accuracy")
    ax.set_xticks(range(1, 16, 2))
    ax.legend()
    ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()