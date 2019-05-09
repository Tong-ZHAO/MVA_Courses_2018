import numpy as np

from kernels.missmatch import MissMatchKernel
from kernels.spectrum import SpectrumKernel
from kernels.substring import SubStringKernel


def load_spectrum_kernel(dataset, k, weight, df, df_test):
    print(f'Using {k}-spectrum kernel (weight: {weight})')
    try:
        K_train = np.load(f'kernel_data/spectrum2_{k}_{weight}_train_{dataset}.npy', )
        K_test = np.load(f'kernel_data/spectrum2_{k}_{weight}_test_{dataset}.npy')

        print('Loaded kernel weights')
    except IOError:
        print('Failed to load kernel weights')

        kernel = SpectrumKernel(k, weight)
        K_train = kernel.fit(df.seq.values)
        K_test = kernel.predict(df_test.seq.values)
        np.save(f'kernel_data/spectrum2_{k}_{weight}_train_{dataset}.npy', K_train, allow_pickle=False)
        np.save(f'kernel_data/spectrum2_{k}_{weight}_test_{dataset}.npy', K_test, allow_pickle=False)

    return K_train, K_test


def load_missmatch_kernel(dataset, k, nmiss, weight, df, df_test):
    print(f'Using {k}-mismatch kernel (weight: {weight}, nmiss: {nmiss})')
    try:
        K_train = np.load(f'kernel_data/missmatch_{k}_{weight}_{nmiss}_train_{dataset}.npy', )
        K_test = np.load(f'kernel_data/missmatch_{k}_{weight}_{nmiss}_test_{dataset}.npy')

        print('Loaded kernel weights')
    except IOError:
        print('Failed to load kernel weights')

        kernel = MissMatchKernel(k, nmiss, weight)
        K_train = kernel.fit(df.seq.values)
        K_test = kernel.predict(df_test.seq.values)

        np.save(f'kernel_data/missmatch_{k}_{weight}_{nmiss}_train_{dataset}.npy', K_train, allow_pickle=False)
        np.save(f'kernel_data/missmatch_{k}_{weight}_{nmiss}_test_{dataset}.npy', K_test, allow_pickle=False)

    return K_train, K_test


def load_substring_kernel(dataset, k, jumps, weight, df, df_test):
    print(f'Using {k}-substring kernel (weight: {weight}, jump: {jumps})')
    try:
        K_train = np.load(f'kernel_data/substring2_{k}_{weight}_{jumps}_train_{dataset}.npy', )
        K_test = np.load(f'kernel_data/substring2_{k}_{weight}_{jumps}_test_{dataset}.npy')

        print('Loaded kernel weights')
    except IOError:
        print('Failed to load kernel weights')

        kernel = SubStringKernel(k, jumps, weight)
        K_train = kernel.fit(df.seq.values)
        K_test = kernel.predict(df_test.seq.values)

        np.save(f'kernel_data/substring2_{k}_{weight}_{jumps}_train_{dataset}.npy', K_train, allow_pickle=False)
        np.save(f'kernel_data/substring2_{k}_{weight}_{jumps}_test_{dataset}.npy', K_test, allow_pickle=False)

    return K_train, K_test
