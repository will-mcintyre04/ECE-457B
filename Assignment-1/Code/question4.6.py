import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def main():
    # Load the dataset and separate features from the target
    df = pd.read_csv("A1Q4-decision-trees.csv")
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # default parameters (gini)
    clf_default = DecisionTreeClassifier(max_depth=3, random_state=123123)
    clf_default.fit(X, y)

    plt.figure(figsize=(15, 10))
    plot_tree(clf_default, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
    plt.title("Default Decision Tree (max_depth=3, Gini)")
    plt.show()

    # change entropy
    clf_entropy = DecisionTreeClassifier(max_depth=3, random_state=123123, criterion="entropy")
    clf_entropy.fit(X, y)

    plt.figure(figsize=(15, 10))
    plot_tree(clf_entropy, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
    plt.title("Entropy Decision Tree")
    plt.show()

    # random splitter
    clf_random = DecisionTreeClassifier(max_depth=3, random_state=123123, splitter="random")
    clf_random.fit(X, y)

    plt.figure(figsize=(15, 10))
    plot_tree(clf_random, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
    plt.title("Random Splitter Decision Tree")
    plt.show()

if __name__ == "__main__":
    main()