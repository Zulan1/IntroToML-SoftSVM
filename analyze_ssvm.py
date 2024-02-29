import numpy as np
import matplotlib.pyplot as plt
from softsvm import softsvm
import time

ROUND_DIGITS = 5

def predict_calculate_error(w, testX, testy):
    """predicts the testy values and calculates the error"""
    y_preds = np.array([np.sign(x @ w) for x in testX])
    return np.mean(np.vstack(testy) != np.vstack(y_preds))

def test_input_size(m: int = 200, l: int = 1):
    """tests the ssvm algorithm for a given sample size and l value"""
    # load question 2 data
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    w = softsvm(l, _trainX, _trainy)

    test_error = predict_calculate_error(w, testX, testy)
    train_error = predict_calculate_error(w, _trainX, _trainy)

    return test_error, train_error

def analyze_lambda_values(sample_size, test_rep, l_values):
    """tests the knn algorithm for different k"""
    averages = []
    errors = []
    for l in l_values:
        test_sample_results, train_sample_results = zip([test_input_size(sample_size, l) for _ in range(test_rep)])
        low = round(min(sample_results), ROUND_DIGITS)
        high = round(max(sample_results), ROUND_DIGITS)
        average = round(sum(sample_results) / test_rep, ROUND_DIGITS)
        averages.append(average)
        errors.append((average - low, high - average))  # error below and above the average

    # Convert errors to a format suitable for error bars (separate positive and negative)
    errors_below, errors_above = zip(*errors)
    yerr = [errors_below, errors_above]
    title = f'Error Range vs. λ Value\n (Sample Size: {sample_size}) (Test Repetitions: {test_rep})'
    x_label = 'λ Value'
    plot_error_bar_graph(l_values, averages, yerr, title, x_label, errors_below, errors_above, 'log')
    
def Q1():
    """tests the knn algorithm for different sample sizes"""
    analyze_lambda_values(100, 10, [10**n for n in range(1, 11)])
    
def Q2():
    """tests the knn algorithm for different sample sizes"""
    analyze_lambda_values(1000, 1, [1, 3, 5, 8])
    

def plot_error_bar_graph(x_values, averages, yerr, title, x_label, errors_below, errors_above, x_scale='linear'):
    """plots error bar graph with the given parameters"""
    global end # pylint: disable=w0601
    # Plotting the graph with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, averages,
                 yerr=yerr, fmt='o',
                 color='b', ecolor='gray',
                 capsize=5, label='Average with Error',
                 )

    # Annotating each data point
    for i, size in enumerate(x_values):
        plt.annotate(f'High: {round(errors_above[i] + averages[i], ROUND_DIGITS)}\n'
                    f'Avg: {round(averages[i], ROUND_DIGITS)}\n'
                    f'Low: {round(averages[i] - errors_below[i], ROUND_DIGITS)}',
                    (size, averages[i]),
                    textcoords="offset points",
                    xytext=(15, 0),
                    ha='center',
                    fontsize=12)

    plt.title(title, fontsize=26)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.xscale(x_scale)
    plt.xticks(x_values, fontsize=12)
    plt.grid(True)
    end = time.time()
    # plt.savefig(f'.\\Graphs\\{title}.png')
    plt.show(block=True)

if __name__ == '__main__':
    # analyze_sample_sizes()
    start = time.time()
    Q1()
    # Q2()
    print(f'time_to_process in seconds {end - start}')
