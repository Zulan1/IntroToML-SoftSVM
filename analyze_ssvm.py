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

def analyze_lambda_values(sample_size, test_rep, l_values, error_bar):
    """tests the knn algorithm for different k"""

    train_averages, train_errors, test_averages, test_errors = [], [], [], []
    def calc_error_params(sample_results, averages: list, errors: list):
        low = round(min(sample_results), ROUND_DIGITS)
        high = round(max(sample_results), ROUND_DIGITS)
        average = round(sum(sample_results) / test_rep, ROUND_DIGITS)
        averages.append(average)
        errors.append((average - low, high - average))

    for l in l_values:
        test_sample_results, train_sample_results = zip(*[test_input_size(sample_size, l) for _ in range(test_rep)])
        calc_error_params(test_sample_results, test_averages, test_errors)
        calc_error_params(train_sample_results, train_averages, train_errors)


    title = 'Error Range vs. λ Value)'
    x_label = 'λ Value'
    def plot_graph(errors, averages, color, label):
        # Convert errors to a format suitable for error bars (separate positive and negative)
        errors_below, errors_above = zip(*errors)
        yerr = [errors_below, errors_above]
        if error_bar:
            plot_error_bar_graph(l_values, averages, yerr, title, x_label, 'log', color=color, label=label)
        else:
            plt.plot(l_values, averages, '^', color=color, label=label, markersize=10)
            # for i, size in enumerate(l_values):
            #     plt.annotate(round(averages[i], ROUND_DIGITS),
            #                 (size, averages[i]),
            #                 textcoords="offset points",
            #                 xytext=(0, 10),
            #                 ha='center',
            #                 fontsize=12)

    color = 'blue' if error_bar else 'orange'
    label = 'Test Sample Size=100' if error_bar else 'Test Sample Size=1000'
    plot_graph(test_errors, test_averages, color=color, label=label)
    color = 'green' if error_bar else 'red'
    label = 'Train Sample Size=100' if error_bar else 'Train Sample Size=1000'
    plot_graph(train_errors, train_averages, color=color, label=label)
    plt.legend(loc='upper left')

def plot_error_bar_graph(x_values, averages, yerr, title, x_label, x_scale='linear', color='b', label='error bar'):
    """plots error bar graph with the given parameters"""
    # Plotting the graph with error bars
    plt.errorbar(x_values, averages,
                 yerr=yerr, fmt='-o',
                 color=color, ecolor='gray',
                 capsize=5, label=label,
                 )

    # Annotating each data point
    # for i, size in enumerate(x_values):
    #     plt.annotate(round(averages[i], ROUND_DIGITS),
    #                 # f'High: {round(errors_above[i] + averages[i], ROUND_DIGITS)}\n'
    #                 # f'Avg: {round(averages[i], ROUND_DIGITS)}\n'
    #                 # f'Low: {round(averages[i] - errors_below[i], ROUND_DIGITS)}',
    #                 (size, averages[i]),
    #                 textcoords="offset points",
    #                 xytext=(0, 10),
    #                 ha='center',
    #                 fontsize=12)

    plt.title(title, fontsize=26)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.xscale(x_scale)
    plt.xticks(x_values, fontsize=12)
    plt.grid(True)

if __name__ == '__main__':
    start = time.time()
    analyze_lambda_values(100, 10, [10 ** n for n in range(1, 11)], True)
    analyze_lambda_values(1000, 1, [10 ** n for n in [1, 3, 5, 8]], False)
    end = time.time()
    plt.show()
    print(f'time_to_process in seconds {end - start}')
