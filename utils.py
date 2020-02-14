import os
import subprocess
import numpy as np

def call_cmd(cmd):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, startupinfo=startupinfo)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return output

def dim(array):
    '''
    Calculate the dimension of a python list 
    Parameter:
    array:
        The python list of which the dimension is calculated from
    '''
    if not type(array) == list:
        return []
    return [len(array)] + dim(array[0])

def get_templates(values):
    if len(values.shape) == 0:
        templates = list(values)
    elif len(values.shape) == 1:
        templates = list(np.sort(np.unique(values)))
    elif len(values.shape) == 2:
        templates = [list(np.sort(np.unique(values[:, i]))) for i in range(values.shape[1])]
    return templates

def integerize(values):
    def integerize_list(values):
        try:
            value_int = {value: index for index, value in enumerate(np.unique(values))}
            result = [value_int[value] for value in values]
        except KeyError:
            print(values)
        
        return result, value_int
        
    if len(dim(values)) == 0:
        raise ValueError("The input variable must be a list of 1d or 2d")
    elif len(dim(values)) == 1:
        return integerize_list(values)
    elif len(dim(values)) == 2:
        # Transpose values variable
        values_T = list(zip(*values))
        
        results = []        
        value_ints = []
        for value in values_T:
            result, value_int = integerize_list(value)
            results.append(result)
            value_ints.append(value_int)
        results = np.array(results).T
        
        return results, value_ints

    
def one_hot_encoding(values, templates, binarize=True):
    def one_hot_encoding_1d(values, template, binarize):
        '''
        Parameter:
        values: numpy array
            A 1d numpy array needed to be one hot encoded
        template: list
            A list specify the template of encoding
        binarize: boolean
            When there are only two distinct values in values variable,
            indcate whether values should be cosidered as 
            binary values (1 or 0) or multiple values (10 or 01) 
            
        Return: numpy array
            A 1d one hot encoded numpy array
        '''
        values = np.array(values)
        template = np.array(template)
        
        # If the template has less than two values
        if binarize and template.shape[0] < 3:
            results = np.array([np.where(template == value)[0][0] for value in values])
        else:            
            results = np.zeros((values.shape[0], template.shape[0]), dtype=int)
            for value, result in zip(values, results):
                index = np.where(template == value)[0][0]
                result[index] = 1
            results = results.reshape((-1))
        return results
        
    if len(values.shape) == 0:
        # values variable is an integer value
        # return a 1d array
        
        # template variable must be a 1d array if the values is an integer
        if len(dim(templates)) != 1:
            raise ValueError('The templates variable must be a 1d array if the values variable is an integer')
        
        results = one_hot_encoding_1d(np.array([values]), templates, binarize)
            
    elif len(values.shape) == 1:
        # values variable is a 1d array
        # return a 1d array
        
        # template variable is a 1d array
        if len(dim(templates)) != 1:
            raise ValueError('The templates variable must be a 1d array if the values variable is a 1d array')

        results = one_hot_encoding_1d(values, templates, binarize)
            
    elif len(values.shape) == 2:
        # values variable is a 2d array
        # return a 2d array
        # assume to one hot encoding for each column of values variable
        
        # template variable is a 2d array. Each row of template variable is for each column of values variable
        if len(dim(templates)) != 2:
            raise ValueError('The templates variable must be a 2d array if the values variable is a 2d array')
        
        results = np.zeros((values.shape[0], 0), dtype=int)
        for value, template in zip(values.T, templates):
            num_col = len(template)
            if binarize and num_col < 3:
                num_col = 1
                
            results = np.concatenate((results, one_hot_encoding_1d(value, template, binarize)
                                      .reshape((values.shape[0], num_col))), axis=1)
        
    return results
