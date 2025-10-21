#!/usr/bin/env python3
"""
Shopping - CS50's Introduction to Artificial Intelligence with Python

Write an AI to predict whether online shopping customers will complete a purchase.
"""

import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an integer (0 for Jan, 1 for Feb, ..., 11 for Dec)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer (0 for non-returning, 1 for returning)
        - Weekend, an integer (0 for non-weekend, 1 for weekend)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    
    # Month mapping
    month_map = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert each row to evidence list
            evidence_row = [
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                month_map[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ]
            evidence.append(evidence_row)
            
            # Convert Revenue to label
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)
    
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
    
    # Calculate sensitivity (true positive rate)
    total_positives = true_positives + false_negatives
    sensitivity = true_positives / total_positives if total_positives > 0 else 0
    
    # Calculate specificity (true negative rate)
    total_negatives = true_negatives + false_positives
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0
    
    return sensitivity, specificity


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=0.4
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {sum(1 for actual, pred in zip(y_test, predictions) if actual == pred)}")
    print(f"Incorrect: {sum(1 for actual, pred in zip(y_test, predictions) if actual != pred)}")
    print(f"True Positive Rate: {sensitivity:.2%}")
    print(f"True Negative Rate: {specificity:.2%}")


if __name__ == "__main__":
    main()
