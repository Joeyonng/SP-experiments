import csv
import random
import numpy as np
from collections import defaultdict
from bisect import bisect_left
from sklearn.model_selection import KFold

from utils import integerize, get_templates, one_hot_encoding

def divide_value(x, thresholds, name, null=None):
    if null is not None and x in null:
        return null[x]
        
    thresholds = sorted(thresholds)

    maps = "".join(list(map(lambda t: name + "<=" + str(t) + "," + str(t) + "<", thresholds)) + [name]).split(",")
    index = bisect_left(thresholds, x)
            
    return maps[index]

def dataset_config(name, dir_path='./data'):
    if(name == 'zoo'):
        path = dir_path + '/zoo.data'
        label_index = 17
        header = {1: 'hair', 2: 'features', 3: 'eggs',  4: 'milk', 5: 'airborne', 6: 'aquatic', 7: 'predator', 8: 'toothed', 
                  9: 'backbone', 10: 'breathes', 11: 'venomous', 12: 'fins', 13: 'legs', 14: 'tail', 15: 'domestic', 16: 'catsize', 17: 'label'},
        transform = {
            1: lambda x: {'1': 'hair', '0': '!hair'}[x],
            2: lambda x: {'1': 'feathers', '0': '!feathers'}[x],
            3: lambda x: {'1': 'eggs', '0': '!eggs'}[x],
            4: lambda x: {'1': 'milk', '0': '!milk'}[x],
            5: lambda x: {'1': 'airborne', '0': '!airborne'}[x],
            6: lambda x: {'1': 'aquatic', '0': '!aquatic'}[x],
            7: lambda x: {'1': 'predator', '0': '!predator'}[x],
            8: lambda x: {'1': 'toothed', '0': '!toothed'}[x],
            9: lambda x: {'1': 'backbone', '0': '!backbone'}[x],
            10: lambda x: {'1': 'breathes', '0': '!breathes'}[x],
            11: lambda x: {'1': 'venomous', '0': '!venomous'}[x],
            12: lambda x: {'1': 'fins', '0': '!fins'}[x],
            13: lambda x: {'0': '0-legs', '2': '2-legs', '4': '4-legs', '5': '5-legs', '6': '6-legs', '8': '8-legs'}[x],
            14: lambda x: {'1': 'tail', '0': '!tail'}[x],
            15: lambda x: {'1': 'domestic', '0': '!domestic'}[x],
            16: lambda x: {'1': 'catsize', '0': '!catsize'}[x],
            17: lambda x: {'1': 'Mammal', '2': 'Bird', '3': 'Reptile', '4': 'Fish', '5': 'Amphibian', '6': 'Bug', '7': 'Invertebrate'}[x]
        }
        hasHeader = False
        unknown = None
    elif(name == 'adult'):
        path = dir_path + '/adult.data'
        label_index = 14
        header = {0: 'age', 1: 'work', 3: 'education', 5: 'martial-status', 6: 'occupation', 7: 'relationship', 8: 'race', 9: 'sex', 
                  10: 'captial-gain', 11: 'captial-loss', 12: 'work-hours', 13: 'country', 14: 'income'}
        transform = {
            0: lambda x: divide_value(int(x), [28, 37, 48], "age"), 
            1: lambda x: {'Private': 'private', 'Self-emp-not-inc': 'self-employed', 'Self-emp-inc': 'self-employed', 
                          'Federal-gov': 'government', 'Local-gov': 'government', 'State-gov': 'government', 
                          'Without-pay': 'no-work', 'Never-worked': 'no-work'}[x], 
            3: lambda x: {'10th': 'dropout', '11th': 'dropout', '12th': 'dropout', '1st-4th':'dropout', '5th-6th': 'dropout', 
                          '7th-8th': 'dropout', '9th': 'dropout', 'Preschool': 'dropout', 'HS-grad': 'high-school-grad',
                          'Some-college': 'high-school-grad', 'Masters': 'college', 'Prof-school': 'prof-school', 
                          'Assoc-acdm': 'associates', 'Assoc-voc': 'associates', 'Bachelors': 'college', 'Doctorate': 'phd'}[x],
            5: lambda x: {'Never-married': 'never-married', 'Married-AF-spouse': 'married','Married-civ-spouse': 'married', 
                          'Married-spouse-absent': 'separated', 'Separated': 'separated', 'Divorced':'separated', 
                          'Widowed': 'widowed'}[x], 
            6: lambda x: {"Adm-clerical": "admin", "Armed-Forces": "military", "Craft-repair": "blue-collar", 
                          "Exec-managerial": "white-collar", "Farming-fishing": "blue-collar", "Handlers-cleaners": "blue-collar", 
                          "Machine-op-inspct": "blue-collar", "Other-service": "service", "Priv-house-serv": "service", 
                          "Prof-specialty": "professional", "Protective-serv": "other", "Sales":"sales", "Tech-support": "other", 
                          "Transport-moving": "blue-collar"}[x], 
            7: lambda x: x,
            8: lambda x: {'White': 'white', 'Asian-Pac-Islander': 'asian-pac-islander', 'Amer-Indian-Eskimo': 'amer-indian-eskimo',
                          'Other': 'other', 'Black': 'black'}[x],
            9: lambda x: {'Female': 'female', 'Male': 'male'}[x], 
            10: lambda x: divide_value(int(x), [0, 7298], "capital-gain"),
            11: lambda x: divide_value(int(x), [0, 1887], "capital-loss"),
            12: lambda x: divide_value(int(x), [40, 45], 'work-hours'),
            13: lambda x: {'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China': 'China',
                           'Columbia': 'South-America', 'Cuba': 'Other', 'Dominican-Republic': 'Latin-America',
                           'Ecuador': 'South-America', 'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
                           'France': 'Euro-1', 'Germany': 'Euro-1', 'Greece': 'Euro-2', 'Guatemala': 'Latin-America',
                           'Haiti': 'Latin-America', 'Holand-Netherlands': 'Euro-1', 'Honduras': 'Latin-America',
                           'Hong': 'China', 'Hungary': 'Euro-2', 'India': 'British-Commonwealth', 'Iran': 'Other',
                           'Ireland': 'British-Commonwealth', 'Italy': 'Euro-1', 'Jamaica': 'Latin-America',
                           'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico': 'Latin-America', 'Nicaragua': 'Latin-America',
                           'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru': 'South-America',
                           'Philippines': 'SE-Asia', 'Poland': 'Euro-2', 'Portugal': 'Euro-2',
                           'Puerto-Rico': 'Latin-America', 'Scotland': 'British-Commonwealth', 'South': 'Euro-2',
                           'Taiwan': 'China', 'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
                           'United-States': 'United-States', 'Vietnam': 'SE-Asia'}[x],
            14: lambda x: x
        }
        hasHeader = True
        unknown = '?'
    elif(name == 'recidivism'):
        path = dir_path + '/recidivism.data'
        label_index = 16
        header = {0: 'color', 1: 'alcohol', 2: 'drugs', 3: 'supervised', 4: 'married', 5: 'felony', 6: 'work-release', 7: 'property', 8: 'person',
                  9: 'sex', 10: 'priors', 11: 'school-years', 12: 'violations', 13: 'age', 14: 'month-in-prison', 16: 'recidivism'}
        transform = {
            0: lambda x: {'1': 'white', '0': 'black'}[x],
            1: lambda x: {'1': 'alcohol', '0': '!alcohol'}[x],
            2: lambda x: {'1': 'drugs', '0': '!drugs'}[x],
            3: lambda x: {'1': 'supervised', '0': '!supervised'}[x],
            4: lambda x: {'1': 'married', '0': '!married'}[x],
            5: lambda x: {'1': 'felony', '0': '!felony'}[x],
            6: lambda x: {'1': 'work-release', '0': '!work-release'}[x],
            7: lambda x: {'1': 'property', '0': '!property'}[x],
            8: lambda x: {'1': 'person', '0': '!person'}[x],
            9: lambda x: {'1': 'male', '0': 'famale'}[x],
            10: lambda x: divide_value(int(x), [1, 5], 'priors'),
            11: lambda x: divide_value(int(x), [8, 11], 'school-years'),
            12: lambda x: divide_value(int(x), [1, 5], 'violations'),
            13: lambda x: divide_value(int(x) / 12, [21, 26, 33], 'age'),
            14: lambda x: divide_value(int(x), [4, 9, 24], 'month-in-prison'),
            16: lambda x: {'1': 're-arrested', '0': 'no-more-crimes'}[x]
        }
        hasHeader = True
        unknown = '-9'
    elif(name == 'lending'):
        path = dir_path + '/lending.data'
        label_index = 16
        header = {2: 'loan-amount', 12: 'home_ownership', 13: 'income', 16: 'loan_status', 29: 'num-inquiries', 35: 'revolving-utilization', 
                  51: 'FIFO-high', 52: 'FIFO-low', 109: 'num-bankruptcies'}
        transform = {
            2: lambda x: divide_value(int(x), [5400, 10000, 15000], "loan-amount"),
            12: lambda x: {'MORTGAGE': 'mortgage', 'NONE': 'none', 'OTHER': 'other', 'OWN': 'own', 'RENT': 'rent'}[x],
            13: lambda x: divide_value(float(x), [40000, 60000, 82000], 'income'),
            16: lambda x: {'Fully Paid': 'good', 'Charged Off': 'bad', 'Default': 'bad'}[x],
            29: lambda x: divide_value(int(x), [0, 1], 'num-inquiries'),
            35: lambda x: divide_value(float(x[:-1]), [25.2, 49.10, 72.23], 'revolving-utilization'),
            51: lambda x: divide_value(int(x), [649, 699, 749], 'FIFO-high'),
            52: lambda x: divide_value(int(x), [645, 695, 745], 'FIFO-low'),
            109: lambda x: {'0': '0-bankruptcy(s)', '1': '1-bankruptcy(s)', '2': '2-bankruptcy(s)'}[x]
        }
        hasHeader = True
        unknown = ""
    elif(name == 'heloc'):
        path = dir_path + '/heloc_dataset_v1.data'
        label_index = 0
        header = {0: 'label', 1: 'external-risk-estimate', 2: 'M-since-oldest-trade-open', 3: 'M-since-latest-trade-open', 4: 'average-M-in-file', 
                  5: 'num-satisfied-trades', 6: 'num-trades-60+-ever', 7: 'num-trades-90+-ever', 8: '%-trades-no-delq', 9: 'M-since-latest-delq' ,
                  10: 'max-delq-last-12M', 11: 'max-delq-ever', 12: 'num-trades', 13: 'num-trades-open-last-12M', 14: '%-install-trades', 
                  15: 'M-since-latest-inq-excl-7d', 16: 'num-inq-last-6M', 17: 'num-inq-last-6M-excl-7d', 18: 'net-fraction-revolving-burden', 
                  19: 'net-fraction-install-burden', 20: 'num-revolving-trades-w-bal', 21: 'num-install-trades-w-bal', 22: 'num-trades-w-high-util-ratio', 
                  23: '%-trades-w-bal'}
        transform = {
            0: lambda x: {'Bad': 'bad', 'Good': 'good'}[x],
            1: lambda x: divide_value(int(x), [64, 71, 76, 81], "external-risk-estimate", null={-7: "-7", -8: "-8", -9: "-9"}),
            2: lambda x: divide_value(int(x), [92, 135, 264], "M-since-oldest-trade-open", null={-7: "-7", -8: "-8", -9: "-9"}),
            3: lambda x: divide_value(int(x), [20], "M-since-latest-trade-open", null={-7: "-7", -8: "-8", -9: "-9"}),
            4: lambda x: divide_value(int(x), [49, 70, 97], "average-M-in-file", null={-7: "-7", -8: "-8", -9: "-9"}),
            5: lambda x: divide_value(int(x), [3, 6, 13, 22], "num-satisfied-trades", null={-7: "-7", -8: "-8", -9: "-9"}),
            6: lambda x: divide_value(int(x), [2, 3, 12, 13], "num-trades-60+-ever", null={-7: "-7", -8: "-8", -9: "-9"}),
            7: lambda x: divide_value(int(x), [2, 8, 10], "num-trades-90+-ever", null={-7: "-7", -8: "-8", -9: "-9"}),
            8: lambda x: divide_value(int(x), [59, 84, 89, 96], "%-trades-no-delq", null={-7: "-7", -8: "-8", -9: "-9"}),
            9: lambda x: divide_value(int(x), [18, 33, 48], "M-since-latest-delq", null={-7: "-7", -8: "-8", -9: "-9"}),
            10: lambda x: divide_value(int(x), [6, 7], "max-delq-last-12M", null={-7: "-7", -8: "-8", -9: "-9"}),
            11: lambda x: divide_value(int(x), [3], "max-delq-ever", null={-7: "-7", -8: "-8", -9: "-9"}),
            12: lambda x: divide_value(int(x), [1, 10, 17, 28], "num-trades", null={-7: "-7", -8: "-8", -9: "-9"}),
            13: lambda x: divide_value(int(x), [3, 4, 7, 12], "num-trades-open-last-12M", null={-7: "-7", -8: "-8", -9: "-9"}),
            14: lambda x: divide_value(int(x), [36, 47, 58, 85], "%-install-trades", null={-7: "-7", -8: "-8", -9: "-9"}),
            15: lambda x: divide_value(int(x), [1, 2, 9, 23], "M-since-latest-inq-excl-7d", null={-7: "-7", -8: "-8", -9: "-9"}),
            16: lambda x: divide_value(int(x), [2, 5, 9], "num-inq-last-6M", null={-7: "-7", -8: "-8", -9: "-9"}),
            17: lambda x: divide_value(int(x), [3], "num-inq-last-6M-excl-7d", null={-7: "-7", -8: "-8", -9: "-9"}),
            18: lambda x: divide_value(int(x), [15, 38, 73], "net-fraction-revolving-burden", null={-7: "-7", -8: "-8", -9: "-9"}),
            19: lambda x: divide_value(int(x), [36, 71], "net-fraction-install-burden", null={-7: "-7", -8: "-8", -9: "-9"}),
            20: lambda x: divide_value(int(x), [4, 5, 8, 12], "num-revolving-trades-w-bal", null={-7: "-7", -8: "-8", -9: "-9"}),
            21: lambda x: divide_value(int(x), [3, 4, 12, 14], "num-install-trades-w-bal", null={-7: "-7", -8: "-8", -9: "-9"}),
            22: lambda x: divide_value(int(x), [2, 3, 4, 6], "num-trades-w-high-util-ratio", null={-7: "-7", -8: "-8", -9: "-9"}),
            23: lambda x: divide_value(int(x), [48, 67, 74, 87], "%-trades-w-bal", null={-7: "-7", -8: "-8", -9: "-9"}),
        }
        hasHeader = True
        unknown = '-9'
    else:
        raise ValueError('Dataset not found')
        
    return path, label_index, header, transform, hasHeader, unknown


class Dataset:
    def __init__(self, path, label_index, header, transform, hasHeader=False, unknown=None):
        '''
        Parameters:
        path: String
            The path to the dataset file
        label_index: int
            The label index of the dataset
        header: dict
            The key of the dict is the column index and the value is the description of the column
        transform: Map
            How to transform the features and labels. The key of the map should be index and the value 
            is a function of the transformation. The return value of the transformation function is 
            expected to be a description. 
        hasHeader: bool
            Indicate whether the first row of the file is the header row
        unknown: String
            How is unknown value represented in the data file
        
        Return: 
        None
        '''
        
        try:
            self.header = [header[index] for index in transform]
        except KeyError:
            print('The specified header contains missing value required by transform function')
            print(KeyError)

        # Transform features and labels
        self.features = []
        self.labels = []
        self.variables = []
        with open(path, errors='ignore') as csvfile:
            file = csv.reader(csvfile, skipinitialspace=True)
            if hasHeader:
                header = next(file)
            
            for row in file:
                # The row is invalid if the length of the row != length of header
                if hasHeader and len(row) != len(header):
                    continue

                feature = []
                for index in transform:
                    error = False
                    if unknown != None and unknown == row[index]:
                        error = True
                        break
        
                    try:
                        transformed = transform[index](row[index].strip())
                        error = False
                    except KeyError: 
                        error = True
                        break
                        
                    if index == label_index:
                        label = transformed
                    else:
                        feature.append(transformed)
                                            
                if(error):
                    continue
                
                self.features.append(feature)
                self.labels.append(label)
                self.variables.append(feature + [label])
        
        # Integerize each variable
        self.V, variable_maps = integerize(self.variables)
        self.X = self.V[:, :-1]
        self.Y = self.V[:, -1]
        
        # Get string representation for ceah variable
        self.V_strs = [sorted(variable_map, key=variable_map.get) for variable_map in variable_maps]
        self.X_strs = self.V_strs[:-1]
        self.Y_strs = self.V_strs[-1]
        
        # Get number of features for each variable
        self.V_nums = np.array(list(map(len, self.V_strs)))
        self.X_nums = self.V_nums[:-1]
        self.Y_nums = self.V_nums[:-1]
        
        # Binarize the features and labels
        self.bX = one_hot_encoding(self.X, get_templates(self.X))
        self.bY = one_hot_encoding(self.Y.reshape(-1, 1), get_templates(self.Y.reshape(-1, 1)), binarize=False)
        self.bX_strs = []
        for X_feature in self.X_strs:
            if len(X_feature) > 2:
                self.bX_strs.extend(X_feature)
            elif len(X_feature) == 2:
                self.bX_strs.append(X_feature[1])
            else:
                raise str(X_feature) + " has only 1 feature"
        self.bY_strs = self.Y_strs
                
               
    def divide_dataset(self, percentage, random=False):
        '''
        Divide the whole dataset into training set and test set.
        
        Parameters:
        percentage: float
            The first percentage of the dataset is training set and rest is test set.
        random: bool
            If True, the dataset will be first shuffled and then splitted.
        Return: 
        None
        '''
        indices = np.arange(self.V.shape[0])
        if random: 
            indices = np.random.choice(indices, indices.shape[0])
            
        train_indices = sorted(indices[:int(percentage * self.V.shape[0])])
        test_indices = sorted(indices[int(percentage * self.V.shape[0]):])

        self.get_dataset(train_indices, test_indices)
        return train_indices, test_indices

    def balance_dataset(self, percentage, random=False):
        '''
        Divide the whole dataset into training set and test set. 
        The training set is balanced: each label has same number of training instances.
        
        Parameters:
        percentage: float
            The first percentage of the dataset is training set and rest is test set.
        random: bool
            If True, the dataset will be first shuffled and then splitted.
        Return: 
        None
        '''
        y_indices = []
        min_count = np.iinfo(np.int32).max
        for y in np.unique(self.Y):
            indices = np.where(self.Y == y)[0]
            min_count = min(indices.shape[0], min_count)
    
            y_indices.append(indices)

        
        train_indices = np.array([]).astype(int)
        test_indices = np.array([]).astype(int)
        for indices in y_indices:
            if random:
                indices = np.random.choice(indices, indices.shape[0])
            train_indices = np.append(train_indices, indices[:int(percentage * min_count)])
            test_indices = np.append(test_indices, indices[int(percentage * min_count):min_count])
        
        self.get_dataset(train_indices, test_indices)
        return train_indices, test_indices
        
    def k_fold_dataset(self, k, random=False):
        y_indices = []
        for y in np.unique(self.Y):
            indices = np.where(self.Y == y)[0]
            if random:
                indices = np.random.choice(indices, indices.shape[0])
            y_indices.append(indices)
        
        balance_indices = np.array([]).astype(int)
        for indices in zip(*y_indices):
            balance_indices = np.append(balance_indices, indices)
            
        k_fold = []
        for train_indices, test_indices in KFold(n_splits=k).split(balance_indices):
            k_fold.append((balance_indices[train_indices], balance_indices[test_indices]))
        return k_fold
        
    def get_dataset(self, train_indices, test_indices):
        self.train_V = self.V[train_indices]
        self.train_X = self.X[train_indices]
        self.train_Y = self.Y[train_indices]
        self.test_V = self.V[test_indices]
        self.test_X = self.X[test_indices]
        self.test_Y = self.Y[test_indices]

        self.train_bX = self.bX[train_indices]
        self.train_bY = self.bY[train_indices]
        self.test_bX = self.bX[test_indices]
        self.test_bY = self.bY[test_indices]
        