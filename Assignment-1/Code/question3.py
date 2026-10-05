import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def euclidean_distance(vector_1, vector_2):
    difference = vector_1 - vector_2
    return np.sum(difference**2)**(1/2)

def cosine_similarity(vector_1, vector_2):
    dot_product = np.dot(vector_1, vector_2)
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    cos_theta = dot_product/(norm_1*norm_2)
    return cos_theta

class myKNN:
    def __init__(self, k, metric='euclidian'):
        self.k = k
        self.metric = metric

    def set_validation_data(self, X_validation, Y_validation):
        self.X_validation = np.array(X_validation)
        self.Y_validation = np.array(Y_validation)
    
    def set_train_data(self, X_train, Y_train):
        # store the training data in our case:
        # (glucose concentration, BMI, age)
        self.X_train = np.array(X_train)
        # note this is the target label
        self.Y_train = np.array(Y_train)
    
    def predict(self, X):
        # predict a target for each point in the set X using
        # the training data
        predictions = [self._predict(x) for x in np.array(X)]
        return np.array(predictions)
    
    def _predict(self, x):
        # predict the expected target using knn for the point x

        # calculate distances based on test point x for all training points
        if self.metric == "Euclidean":
            distances = [euclidean_distance(v, x) for v in self.X_train]
            # get the k lowest distances
            k_indices = np.argsort(distances)[:self.k]
        elif self.metric == "Cosine":
            distances = [cosine_similarity(v, x) for v in self.X_train]
            # cosine is similarity metric: higher is better, take last elements
            # and reverse them
            k_indices = np.argsort(distances)[-self.k:][::-1]
        else:
            raise ValueError("Unsupported metric. " \
            "Can only use 'Euclidean' or 'Cosine'")

        # get the most common labels for the k nearest neighbours:
        labels = self.Y_train[k_indices]
        counter = Counter(labels)
        most_common_target = counter.most_common(1)[0][0]

        return most_common_target
    
    def evaluate_performance(self, X_test, Y_test):
        # evaluation function to measure accuracy for validation/test sets
        training_predictions = self.predict(self.X_train)
        training_accuracy = np.mean(training_predictions == self.Y_train)
        training_error = 1 - training_accuracy

        validation_predictions = self.predict(self.X_validation)
        validation_accuracy = np.mean(validation_predictions == 
                                      self.Y_validation)
        validation_error = 1 - validation_accuracy

        test_predictions = self.predict(X_test)
        test_accuracy = np.mean(test_predictions == Y_test)
        test_error = 1 - test_accuracy
        
        return (training_accuracy, training_error, validation_accuracy, 
                validation_error, test_accuracy, test_error)

def run_evaluation(X_train, Y_train, X_val, Y_val, X_test, Y_test, k, metric):
    knn = myKNN(k = k, metric=metric)
    knn.set_train_data(X_train, Y_train)
    knn.set_validation_data(X_val, Y_val)

    train_acc, train_err, val_acc, val_err, test_acc, test_err = \
        knn.evaluate_performance(X_test, Y_test)
    print(f"Custom Implementation Metrics for k={k}, distance={metric}:")
    print(f"Training Accuracy: {train_acc:.4f} (Error: {train_err:.4f})")
    print(f"Validation Accuracy: {val_acc:.4f} (Error: {val_err:.4f})")
    print(f"Test Accuracy: {test_acc:.4f} (Error: {test_err:.4f})\n")

def question_3_2(X, Y):
    # The splits required by the assignment
    split_configs = {
        "80/10/10": (0.80, 0.10, 0.10),
        "34/33/33": (0.34, 0.33, 0.33),
        "25/25/50": (0.25, 0.25, 0.50)
    }
    metrics = ["Euclidean", "Cosine"]
    k = 1 # 1 nearest neighbour for this section
    
    for split_name, (train_r, val_r, test_r) in split_configs.items():
        print(f"\n{'='*30}")
        print(f"=== Data Split: {split_name} ===")
        print(f"{'='*30}")
        
        # 1. Split off the test set first
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y, test_size=test_r, shuffle=False
        )
        
        # 2. Split the remaining data into train and validation
        val_ratio_adjusted = val_r / (train_r + val_r)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=val_ratio_adjusted, shuffle=False
        )
        
        for metric in metrics:
            run_evaluation(X_train, Y_train, X_val, Y_val, X_test, Y_test, k, metric)

            # predict and find accuracy using SkLearn
            sk_knn = KNeighborsClassifier(n_neighbors=k, metric=metric.lower())
            sk_knn.fit(X_train, Y_train)

            sk_train_acc = accuracy_score(Y_train, sk_knn.predict(X_train))
            sk_train_err = 1 - sk_train_acc
            sk_val_acc = accuracy_score(Y_val, sk_knn.predict(X_val))
            sk_val_err = 1 - sk_val_acc
            sk_test_acc = accuracy_score(Y_test, sk_knn.predict(X_test))
            sk_test_err = 1 - sk_test_acc
            print(f"SkLearn Metrics for k={k}, distance={metric}:")
            print(f"Training Accuracy: {sk_train_acc:.4f} (Error: {sk_train_err:.4f})")
            print(f"Validation Accuracy: {sk_val_acc:.4f} (Error: {sk_val_err:.4f})")
            print(f"Test Accuracy: {sk_test_acc:.4f} (Error: {sk_test_err:.4f})\n")
            
def question_3_3(X, Y):
    split_configs = {
        "80/10/10": (0.80, 0.10, 0.10),
        "34/33/33": (0.34, 0.33, 0.33), 
        "25/25/50": (0.25, 0.25, 0.50)
    }
    metrics = ["Euclidean", "Cosine"]
    k_values = [1, 3, 5, 11]
    
    # Set up a 3x2 grid for plotting (3 splits, 2 metrics)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Store the best configuration found
    best_error = float('inf')
    best_config = {}

    for i, (split_name, (train_r, val_r, test_r)) in enumerate(split_configs.items()):
        # Split the data using a fixed seed so the graphs are reproducible
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y, test_size=test_r, shuffle=False
        )
        val_ratio_adjusted = val_r / (train_r + val_r)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=val_ratio_adjusted, shuffle=False
        )
        
        for j, metric in enumerate(metrics):
            train_errors = []
            val_errors = []
            test_errors = []
            
            for k in k_values:
                custom_knn = myKNN(k=k, metric=metric)
                custom_knn.set_train_data(X_train, Y_train)
                custom_knn.set_validation_data(X_val, Y_val)
                
                # Fetch errors using your pre-built evaluation method
                train_acc, train_err, val_acc, val_err, test_acc, test_err = custom_knn.evaluate_performance(X_test, Y_test)
                
                train_errors.append(train_err)
                val_errors.append(val_err)
                test_errors.append(test_err)
                
                # Track the best performing model based on validation error
                if val_err < best_error:
                    best_error = val_err
                    best_config = {
                        'k': k, 
                        'metric': metric, 
                        'split': split_name, 
                        'train_err': train_err,
                        'val_err': val_err, 
                        'test_err': test_err
                    }

            # Plotting the three error lines for this specific split and metric
            ax = axes[i, j]
            ax.plot(k_values, train_errors, label='Train Error', marker='o', linestyle='-')
            ax.plot(k_values, val_errors, label='Val Error', marker='s', linestyle='--')
            ax.plot(k_values, test_errors, label='Test Error', marker='^', linestyle=':')
            
            ax.set_title(f"Split: {split_name} | Metric: {metric}")
            ax.set_xlabel("Number of Neighbors (k)")
            ax.set_ylabel("Error Rate")
            
            ax.set_xticks(k_values)
            ax.locator_params(axis='y', nbins=15)
            ax.legend()
            ax.grid(True, which='both', alpha=0.5) 

    plt.suptitle("KNN Error Rates Across Splits, Metrics, and k-values", fontsize=16)
    plt.show()

    print(f"\n{'='*40}")
    print("=== BEST COMBINATION IDENTIFIED ===")
    print(f"{'='*40}")
    print(f"Split:            {best_config['split']}")
    print(f"Metric:           {best_config['metric']}")
    print(f"k-value:          {best_config['k']}")
    print(f"Training Error:   {best_config['train_err']:.4f}")
    print(f"Validation Error: {best_config['val_err']:.4f}")
    print(f"Test Error:       {best_config['test_err']:.4f}")
    
def main():
    data = np.loadtxt('A1Q4NearestNeighbors.csv', delimiter=',')
    X = data[:, :3] # all rows, first 3 columns
    Y = data[:, 3] # all rows, 4th column (index 3)

    ## COMMENT OUT DESIRED SECTION
    #question_3_2(X, Y)
    question_3_3(X, Y)

if __name__ == "__main__":
    main()
