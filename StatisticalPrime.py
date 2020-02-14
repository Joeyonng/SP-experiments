import os
import random
import string
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils import one_hot_encoding, get_templates, call_cmd

class StatisticalPrime:
    def __init__(self, dataset):
        self.dataset = dataset
        # Count how many labels a set of unique features has
        X_Y_count = defaultdict(lambda: defaultdict(int))
        for v in dataset.train_V:
            X_Y_count[tuple(v[:-1])][v[-1]] += 1
            
        # A set of unique features can only have one unique label
        unique_V = []
        for x, Y_count in X_Y_count.items():
            # Select the majority label
            y = sorted(list(Y_count.items()), key=lambda x: x[1], reverse=True)[0][0]
            v = list(x) + [y]
            unique_V.append(v)
        self.unique_pV = np.power(2, unique_V)
        
        # Precompute variables to speed up in finding statistaical primes
        self.train_pV = np.power(2, dataset.train_V)
        self.X_bin_nums = [np.flip(np.power(2, np.arange(X_num))) for X_num in dataset.X_nums]

        self.pure_primes = None
        self.stat_primes = None
        self.stat_statis = None
        
    def espresso_primes(self, cmd="espresso.exe -D expand "):
        '''
        Call espresso program to derive a set of pure prime implicants from self.unique_pV.
        This function will write the dataset into a random file for espresso to read and will read the 
        results from espresso into a numpy array.
        
        Parameters:
        cmd: String
        The command to execute to run espresso program. The default value of this variable assumes that
        1. the espresso program is present in the current folder
        2. the espresso program is the windows version
        3. the espresso program has the "expand" subprogram
        
        Return: 
        pure_primes: numpy array
        A 2d numpy array of pure primes
        '''
        # Randomly generate a temporary Espresso input file name
        file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=5)) + '.pla'
        
        with open(file_name, "w") as file:
            file.write(".mv " + str(self.dataset.V_nums.shape[0]) + " 0 ")
            file.write("".join([str(length) + " " for length in self.dataset.V_nums]))
            file.write("\n.type fr\n")

            for x, y in zip(self.unique_pV[:, :-1], self.unique_pV[:, -1]):
                # Write multi value part
                file.write("".join(["|" + bin(f)[2:].zfill(length) for f, length in zip(x, self.dataset.V_nums[:-1])]))

                # Write label part
                file.write("|" + bin(y)[2:].zfill(self.dataset.V_nums[-1]) + "\n")

            file.write(".e\n")            

        # Call espresso to compute primes
        cmd = cmd + file_name
        output = call_cmd(cmd).decode("utf-8")
        
        # Whether Espresso is successful
        if output[0] != 'e':
            expressions = list(filter(lambda x: not x.startswith("."), output.strip().split("\r\n")))
            os.remove(file_name)

            pure_primes = []
            for expression in expressions:
                variable_primes = expression.strip().split(" ")
                pure_primes.append(tuple(map(lambda x: int(x, 2), variable_primes)))
            self.pure_primes = np.array(pure_primes)
        else:
            print('Espresso run is not successful! Here is the output:')
            print(output)

        print("pure_primes shape:", self.pure_primes.shape)
        return self.pure_primes
    
    def statistical_primes(self, iterations=20, min_cov=0.05):
        '''
        Algorithm to derive statistical primes from self.pure_primes
        
        Parameters: 
        iterations: int
        The number of iterations for each pure prime
        min_cov: float
        The minimum coverage for early stop
        
        Return:
        stat_primes: the 2d numpy array of statistical primes
        stat_statis: the statistics associated with the SPs
        '''
        STATI_INDEX = {'coverage': 0, 'precision': 1, 'density_m': 2, 'density_n': 3}

        cand_primes = np.zeros((0, len(self.dataset.V_nums)), dtype=int)
        cand_statis = np.zeros((0, len(STATI_INDEX)), dtype=float)
        for pure_prime in tqdm(self.pure_primes):
            sibling_primes = np.zeros((0, cand_primes.shape[1]), dtype=int)
            sibling_statis = np.zeros((0, cand_statis.shape[1]), dtype=float)
            for i in range(iterations):
                # Starting point is all don't care (nothing) except that the label is pure prime's label
                cand_prime = np.append(np.power(2, self.dataset.X_nums) - 1, pure_prime[-1])

                # Add literals until it is eqaul to the prime
                while (cand_prime != pure_prime).any():
                    temp_primes = []
                    temp_scores = []
                    temp_statis = []

                    diff_prime = cand_prime[:-1] - pure_prime[:-1]
                    for index, (diff_var, stat_var, X_bin_num) in enumerate(zip(diff_prime, cand_prime, self.X_bin_nums)):
                        # Get the indices of literals in variable of diff_prime that are 1
                        one_indices = (diff_var // X_bin_num % 2 == 1).nonzero()

                        # Flip the 1 literals to 0 for statistical prime
                        candidates = stat_var - X_bin_num[one_indices]

                        # Get all possible next-level statistcal primes and calculate their scores
                        for candidate in candidates:
                            temp_prime = cand_prime.copy()
                            temp_prime[index] = candidate

                            temp_stati = StatisticalPrime.statistic(temp_prime, self.train_pV, self.dataset.V_nums)
                            temp_score = temp_stati[STATI_INDEX['precision']]

                            temp_primes.append(temp_prime)
                            temp_statis.append(temp_stati)
                            temp_scores.append(temp_score)
                    temp_primes = np.array(temp_primes)
                    temp_scores = np.array(temp_scores)
                    temp_statis = np.array(temp_statis)

                    # Random pick one according to the scores
                    index = np.random.choice(np.arange(temp_scores.shape[0]), p=temp_scores / np.sum(temp_scores))
                    cand_prime = temp_primes[index]
                    cand_stati = temp_statis[index]
                    if cand_stati[STATI_INDEX['coverage']] < min_cov:
                        break

                    # Prelimiary SP check: drop this prime if it is included by its sibling primes and has relative low precision
                    include_indices = StatisticalPrime.include(cand_prime, sibling_primes)
                    if include_indices.shape[0] != 0:
                        include_statis = sibling_statis[include_indices]
                        if (cand_stati[STATI_INDEX['precision']] < include_statis[:, STATI_INDEX['precision']]).any():
                            continue

                    sibling_primes = np.vstack((sibling_primes, cand_prime))
                    sibling_statis = np.vstack((sibling_statis, cand_stati))
            cand_primes = np.vstack((cand_primes, sibling_primes))
            cand_statis = np.vstack((cand_statis, sibling_statis))
        _, unique_indices = np.unique(cand_primes, axis=0, return_index=True)
        cand_primes = cand_primes[unique_indices]
        cand_statis = cand_statis[unique_indices]

        # Post-mining optimization
        sorted_indices = np.lexsort((cand_statis[:, STATI_INDEX['density_m']], cand_statis[:, STATI_INDEX['density_n']]))

        self.stat_primes = np.zeros((0, cand_primes.shape[1]), dtype=int)
        self.stat_statis = np.zeros((0, cand_statis.shape[1]), dtype=float)
        for index in tqdm(sorted_indices):
            include_indices = StatisticalPrime.include(cand_primes[index], self.stat_primes)
            if include_indices.shape[0] != 0:
                include_statis = self.stat_statis[include_indices]
                if (cand_statis[index][STATI_INDEX['precision']] < include_statis[:, STATI_INDEX['precision']]).any():
                    continue
            self.stat_primes = np.vstack((self.stat_primes, cand_primes[index]))
            self.stat_statis = np.vstack((self.stat_statis, cand_statis[index]))
        
        print("stat_primes shape:", self.stat_primes.shape)
        return self.stat_primes, self.stat_statis
    
    @staticmethod
    def integerize(prime, lengths, flatten=False):
        if flatten:
            binarys = []
            for length in lengths:
                binarys.append(prime[:length])
                prime = prime[length:]
        else:
            binarys = prime

        result = []
        for binary in binarys:
            integer = int("".join(list(map(str, binary))), 2)
            result.append(integer)

        return result
    
    @staticmethod
    def binarize(prime, lengths, flatten=False):
        result = []
        for var, length in zip(prime, lengths):
            binary = bin(var)[2:].zfill(length)

            # Convert to list and add to result
            binary = list(map(int, list(binary)))
            if flatten:
                result.extend(binary)
            else:
                result.append(binary)

        return result
    
    @staticmethod
    def match(prime, primes):
        '''
        The "statisfy" method in the paper
        
        prime: 1d numpy array
        The test rule
        primes: 2d numpy array
        The instances
        
        Return: 1d numpy array
        Return the index of the instances in primes (instances) that the prime (test rule) satisfies
        '''
        return np.where(np.bitwise_and(prime, primes[:, :prime.shape[0]]).min(axis=1) != 0)[0]
    
    @staticmethod
    def include(prime, primes):
        '''
        The "cover" method in the paper
        
        Parameters:
        prime: 1d numpy array
        The test rule
        primes: 2d numpy array
        The rules
        
        Return: 1d numpy array
        Return the index of the rules in primes (rules) that the prime (test rule) covers
        '''
        return np.where((np.bitwise_and(prime, primes[:, :prime.shape[0]]) == prime).all(axis=1))[0]

    @staticmethod
    def interpret(prime, V_strs):
        '''
        Print the rule in a nice way
        
        Parameters:
        prime: 1d numpy array
        The rule to interpret
        V_strs: 2d list
        From the dataset class
        
        Return: 2d list of strings
        '''
        prime_str = []
        
        binaries = StatisticalPrime.binarize(prime, map(len, V_strs))
        for binary, V_str in zip(binaries, V_strs):
            indices = np.where(np.array(binary) == 1)[0]
            if indices.shape[0] == len(V_str):
                continue
            prime_str.append([list(reversed(V_str))[index] for index in indices])

        return prime_str
    
    @staticmethod
    def statistic(prime, pV, V_nums):
        '''
        Calculate the coverage, precision, density of the prime in the context of pV
        
        Parameters:
        prime: 1d numpy array
        The rule 
        pV: 2d numpy array
        The power of 2 of the dataset
        V_nums: 1d list
        From the dataset
        
        Return: coverage, precision, density_m, density_ns
        '''
        indices = np.where(np.bitwise_and(pV[:, :-1], prime[:-1]).min(axis=1) != 0)[0]
        num_coverage = indices.shape[0]
        coverage = num_coverage / pV.shape[0]
        
        num_precision = np.where(prime[-1] == pV[:, -1][indices])[0].shape[0]
        if num_coverage == 0:
            precision = 0
        else:
            precision = num_precision / num_coverage
            
        care_indices = np.nonzero(prime[:-1] - (np.power(2, V_nums[:-1]) - 1))[0]
        density_m = care_indices.shape[0]
        density_n = np.count_nonzero(StatisticalPrime.binarize(prime[care_indices], V_nums[care_indices], flatten=True))
        
        return coverage, precision, density_m, density_n
        
        