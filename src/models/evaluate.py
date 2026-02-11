def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the performance of the trained model on the test dataset.

    Parameters:
    model: The trained model to evaluate.
    test_data: The data to test the model on.
    test_labels: The true labels for the test data.

    Returns:
    dict: A dictionary containing evaluation metrics such as accuracy, precision, recall, and F1 score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }