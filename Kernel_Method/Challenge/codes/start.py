from time import strftime, time

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from tee import StdoutTee

from algos.svm import SVM
from kernel_io import load_spectrum_kernel, load_substring_kernel, load_missmatch_kernel


def load_data(dataset, train=True):
    letter = 'r' if train else 'e'

    df_Xtr = pd.read_csv(f'data/Xt{letter}{dataset}.csv', index_col=0)

    if train:
        df_Ytr = pd.read_csv(f'data/Ytr{dataset}.csv', index_col=0)

        y = df_Ytr.Bound.values.ravel()
    else:
        y = None

    return df_Xtr, y


def C_to_lambda(C):
    return 1 / (2 * 2000 * C)


def find_hyperparameter(K_train, y_train):
    C_log_min, C_log_max = -3, 0
    space = hp.loguniform('C', C_log_min, C_log_max)
    print(f'Fitting C between {np.exp(C_log_min)} and {np.exp(C_log_max)}')

    start = time()

    def objective(param):
        clf = SVM(lbda=C_to_lambda(param))
        cv = cross_validate(clf, K_train, y_train, scoring='accuracy', cv=5, return_train_score=True, n_jobs=-1)

        train_score = np.mean(cv['train_score'])
        test_score = np.mean(cv['test_score'])
        print(f"{param:.3f}\t{train_score:.3f}\t{test_score:.3f}")

        return dict(
            loss=-test_score,
            status=STATUS_OK,
            extra=dict(
                param=param,
                train_score=train_score,
                test_score=test_score,
            ),
        )

    trials = Trials()
    fmin(objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    best_res = sorted(trials.results, key=lambda x: x['extra']['test_score'], reverse=True)[0]['extra']

    print(best_res)

    print(f'spent {time() - start} s')

    return best_res['param'], best_res['test_score']


def main_for_dataset(dataset):
    df, y_train = load_data(dataset)
    df_test, _ = load_data(dataset, train=False)

    K_train = np.zeros((2000, 2000))
    K_test = np.zeros((1000, 2000))

    # ===== Spectrum Kernel =====
    Ks = {
        '0': [],
        '1': [],
        '2': [12, 15],
    }
    weight = None
    for i in Ks[dataset]:
        K_train_spectrum, K_test_spectrum = load_spectrum_kernel(dataset, i, weight, df, df_test)
        K_train += K_train_spectrum
        K_test += K_test_spectrum

    Ks = {
        '0': [(12, 0.75)],
        '1': [(10, 0.75)],
        '2': [(10, 0.50)],
    }
    for i, weight in Ks[dataset]:
        K_train_spectrum, K_test_spectrum = load_spectrum_kernel(dataset, i, weight, df, df_test)
        K_train += K_train_spectrum
        K_test += K_test_spectrum
    # ===== Spectrum Kernel =====

    # ===== Substring Kernel =====
    Ks = {
        '0': [(8, 2, 0.75)],
        '1': [(12, 2, 0.75)],
        '2': [(12, 2, 0.75)],
    }
    for i, jumps, weight in Ks[dataset]:
        K_train_substring, K_test_substring = load_substring_kernel(dataset, i, jumps, weight, df, df_test)
        K_train += K_train_substring
        K_test += K_test_substring
    # ===== Substring Kernel =====

    # ===== MissMatchKernel Kernel =====
    Km = {
        '0': [(8, 1, 0.75)],
        '1': [(10, 1, 0.50)],
        '2': [],
    }
    for i, mismatches, weight in Km[dataset]:
        K_train_substring, K_test_substring = load_missmatch_kernel(dataset, i, mismatches, weight, df, df_test)
        K_train += K_train_substring
        K_test += K_test
    # ===== MissMatchKernel Kernel =====

    C, test_score = find_hyperparameter(K_train, y_train)

    clf = SVM(lbda=C_to_lambda(C))
    clf.fit(K_train, y_train)

    y_hat = clf.predict(K_train)
    acc = accuracy_score(y_train, y_hat)

    print('Dataset ', dataset, 'train accuracy:', acc)

    y_test = clf.predict(K_test)

    return pd.Series(data=y_test, name='Bound', index=df_test.index), test_score


def main():
    datasets = ['0', '1', '2']
    model_name = 'logs/' + strftime("%Y%m%d_%H%M%S")

    with StdoutTee(f"{model_name}.log", 'w', 1):
        print('Starting...')

        preds = []
        cv_scores = []
        for dataset in datasets:
            pred, cv_score = main_for_dataset(dataset)
            preds.append(pred)
            cv_scores.append(cv_score)

        df = pd.concat(preds, axis=0)

        for i, dataset in enumerate(datasets):
            print(f'Dataset {dataset}: {cv_scores[i]}')
        print(f'Expecting a score of {np.mean(cv_scores)}')

        df.to_csv(f'{model_name}.csv', header=True)


if __name__ == "__main__":
    main()
