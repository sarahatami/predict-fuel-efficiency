# Sarah Hatami


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

random.seed(2)

pd.set_option('display.width', 1000)
pd.set_option('max_columns', 100)
np.set_printoptions(threshold=sys.maxsize)


# get data and make df
autompg_df = pd.read_fwf(r'E:\MASTER\Uni\Term1\ML\hw1\a1_code\auto-mpg.data')  # 391*9
autompg_df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'modelyear',
                      'origin','carname']

def polynomial_regression_all_features():
    # normalization
    numeric_columns=autompg_df.iloc[:, 0:8] #391*8
    normalized_autompg_df = (numeric_columns - numeric_columns.mean()) / numeric_columns.std()
    # define X,Y,train,test
    X=normalized_autompg_df.iloc[:, 1:8] #391*7
    Y=normalized_autompg_df.iloc[:, 0:1] #391*1
    X_train = X[:100] #100*7
    Y_train = Y[:100] #100*1
    X_test = X[100:] #291*7
    Y_test = Y[100:] #291*1

    #polynomial function
    def create_basis_functions(dataset, k): #(1+7*k)
        Z = pd.DataFrame()
        n = 0
        for i in range(1, k + 1):
            for (x, data) in dataset.iteritems():
                Z.insert(n, '%s^%s' % (x, i), (data ** i))
                n += 1
        Z.insert(0, 'bias', 1)
        return Z

    def calculate_error(dataset_x,dataset_y,k):
        Z = create_basis_functions(X_train, k)
        W = np.linalg.inv(Z.values.T @ Z.values) @ Z.values.T @ Y_train.values  # (1+7^k)*1
        # prediction
        predicted_y = np.dot(create_basis_functions(dataset_x, k), W)
        #error
        rmse = np.sqrt(np.sum((predicted_y - dataset_y.values) ** 2) / dataset_y.shape[0])
        return rmse

    #make df of errors
    error_df=pd.DataFrame(columns=['degree','training_error','test_error'])
    for i in range(1,11):
        error_df.loc[i] = [i] + [calculate_error(X_train,Y_train,i)] +[calculate_error(X_test,Y_test,i)]
    #visualization
    plt.plot(error_df['degree'],(error_df['training_error']),marker = ".",color='blue',label='training error')
    plt.plot(error_df['degree'],(error_df['test_error']),marker = ".",color='red',label='test error')
    plt.xlabel('degree')
    plt.ylabel('RMS error')
    plt.xticks(error_df['degree'])
    plt.legend(loc='best')
    plt.show()
    print(error_df)


def polynomial_regression_3rdfeature():
    # define X,Y
    X=autompg_df.iloc[:, 3:4] #391*1
    Y=autompg_df.iloc[:, 0:1] #391*1
    # normalization
    normalized_x= (X - X.mean()) / X.std()
    normalized_y= (Y - Y.mean()) / Y.std()
    # define train and test
    X_train = normalized_x[:100]  # 100*1
    Y_train = normalized_y[:100]  # 100*1
    X_test = normalized_x[100:]  # 291*1
    Y_test = normalized_y[100:]  # 291*1

    #polynomial function
    def create_basis_functions(dataset, k): #*(1+k)
        Z = pd.DataFrame()
        n = 0
        for i in range(1, k + 1):
            for (x, data) in dataset.iteritems():
                Z.insert(n, '%s^%s' % (x, i), (data ** i))
                n += 1
        Z.insert(0, 'bias', 1)
        return Z

    def calculate_predicted_y(dataset_x, k):
        Z = create_basis_functions(X_train, k)
        W = np.linalg.inv(Z.values.T @ Z.values) @ Z.values.T @ Y_train.values  # (1+7*k)*1
        # prediction
        predicted_y = np.dot(create_basis_functions(dataset_x, k), W)
        return predicted_y

    sorted_normalized_x=normalized_x.sort_values("horsepower")
    plt.scatter(X_train, Y_train, color='black', label='training data')
    plt.scatter(X_test, Y_test, color='red', label='tast data')
    plt.plot(sorted_normalized_x,calculate_predicted_y(sorted_normalized_x,1),color='blue',label='predicted mpg')
    plt.xlabel('horsepower')
    plt.ylabel('fuel_efficiency(mpg)')
    plt.legend(loc='best')
    plt.show()

    plt.scatter(X_train, Y_train, color='black', label='training data')
    plt.scatter(X_test, Y_test, color='red', label='tast data')
    plt.plot(sorted_normalized_x,calculate_predicted_y(sorted_normalized_x,5),color='blue',label='predicted mpg')
    plt.xlabel('horsepower')
    plt.ylabel('fuel_efficiency(mpg)')
    plt.legend(loc='best')
    plt.show()

    plt.scatter(X_train,Y_train,color='black',label='training data')
    plt.scatter(X_test,Y_test,color='red',label='tast data')
    plt.plot(sorted_normalized_x,calculate_predicted_y(sorted_normalized_x,10),color='blue',label='predicted mpg')
    plt.xlabel('horsepower')
    plt.ylabel('fuel_efficiency(mpg)')
    plt.legend(loc='best')
    plt.show()


def regularized_polynomial_regression_3rdfeature():
    # define X,Y
    X=autompg_df.iloc[:, 3:4] #391*1
    Y=autompg_df.iloc[:, 0:1] #391*1
    # normalization
    normalized_x= (X - X.mean()) / X.std()
    normalized_y= (Y - Y.mean()) / Y.std()
    # define train and test
    X_train = normalized_x[:100]  # 100*1
    Y_train = normalized_y[:100]  # 100*1
    X_test = normalized_x[100:]  # 291*1
    Y_test = normalized_y[100:]  # 291*1

    def create_basis_functions(dataset): #*9
        Z = pd.DataFrame()
        n = 0
        for i in range(1, 9):
            for (x, data) in dataset.iteritems():
                Z.insert(n, '%s^%s' % (x, i), (data ** i))
                n += 1
        Z.insert(0, 'bias', 1)
        return Z

    def calculate_error(dataset_x, dataset_y, l):
        Z = create_basis_functions(X_train)
        W = np.linalg.inv((Z.values.T @ Z.values) + (l * np.identity(9))) @ Z.values.T @ Y_train.values  # 9*1
        # prediction
        predicted_y = np.dot(create_basis_functions(dataset_x), W)
        #error
        rmse = np.sqrt(np.sum((predicted_y - dataset_y.values) ** 2) / dataset_y.shape[0])
        return rmse

    #make df of errors
    error_df=pd.DataFrame(columns=['lambda','training_error','test_error'])
    for i in (0,0.01,0.1,10,100,1000):
        error_df.loc[i] = [i] + [calculate_error(X_train,Y_train,i)] +[calculate_error(X_test,Y_test,i)]
    print(error_df)

    plt.semilogx(error_df['lambda'],error_df['training_error'],marker = ".",color='blue',label='trainig error')
    plt.semilogx(error_df['lambda'],error_df['test_error'],marker = ".",color='red',label='test error')
    plt.xlabel('lambda')
    plt.ylabel('RMS error')
    plt.legend(loc='best')
    plt.show()


def gaussian_regression_all_features():
    # normalization
    numeric_columns=autompg_df.iloc[:, 0:8] #391*8
    normalized_autompg_df = (numeric_columns - numeric_columns.mean()) / numeric_columns.std()
    # define X,Y,train,test
    X=normalized_autompg_df.iloc[:, 1:8] #391*7
    Y=normalized_autompg_df.iloc[:, 0:1] #391*1
    X_train = X[:100] #100*7
    Y_train = Y[:100] #100*1
    X_test = X[100:] #291*7
    Y_test = Y[100:] #291*1

    #gaussian function
    def create_gaussian_functions(dataset, index_of_mus):
        # std
        s = 2
        #calculate rbf
        def create_rbf(mu):
            def rbf(x):
                phi = np.exp(-np.sum((x - X_train[mu:mu+1].values)**2)/(2*(s**2)))
                return phi
            return rbf

        #create Phi
        list_of_rbfs=[create_rbf(m) for m in index_of_mus]
        Phi=list(map(lambda f:np.apply_along_axis(f,1,dataset),list_of_rbfs))
        #insert bais
        Phi.insert(0, np.repeat(1, np.size(dataset, axis=0)))
        return np.column_stack(Phi)

    def calculate_error(dataset_x,dataset_y,index_of_mus):
        #calculate W and predicte y
        Phi = create_gaussian_functions(X_train, index_of_mus)
        W = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y_train.values  # (num_func+1)*1
        # prediction
        predicted_y = np.dot(create_gaussian_functions(dataset_x, index_of_mus), W)
        # error
        rmse = np.sqrt(np.sum((predicted_y - dataset_y.values) ** 2) / dataset_y.shape[0])
        return rmse

    def choose_means(num, start=0, end=99):
        arr = []
        tmp = random.randint(start, end)
        for x in range(num):
            while tmp in arr:
                tmp = random.randint(start, end)
            arr.append(tmp)
        arr.sort()
        return arr

    #make df of errors
    error_df=pd.DataFrame(columns=['number_of_rbfs','training_error','test_error'])
    list_of_mus=[choose_means(u) for u in [5,15,25,35,45,55,65,75,85,95]]
    for i in range(0,10):
        error_df.loc[i] = [len(list_of_mus[i])] + [calculate_error(X_train,Y_train,list_of_mus[i])] + [calculate_error(X_test,Y_test,list_of_mus[i])]
    plt.plot(error_df['number_of_rbfs'],(error_df['training_error']),marker = ".",color='blue',label='training error')
    plt.plot(error_df['number_of_rbfs'],(error_df['test_error']),marker = ".",color='red',label='test error')
    plt.xlabel('number_of_rbfs')
    plt.ylabel('RMS error')
    plt.xticks(error_df['number_of_rbfs'])
    plt.legend(loc='best')
    plt.show()
    print(error_df)


def regularized_gaussian_regression_all_features():
    # normalization
    numeric_columns=autompg_df.iloc[:, 0:8] #391*8
    normalized_autompg_df = (numeric_columns - numeric_columns.mean()) / numeric_columns.std()
    # define X,Y,train,test
    X=normalized_autompg_df.iloc[:, 1:8] #391*7
    Y=normalized_autompg_df.iloc[:, 0:1] #391*1
    X_train = X[:100] #100*7
    Y_train = Y[:100] #100*1
    X_test = X[100:] #291*7
    Y_test = Y[100:] #291*1
    # gaussian function
    def create_gaussian_functions(dataset, index_of_mus):
        # std
        s = 2
        # calculate rbf
        def create_rbf(mu):
            def rbf(x):
                phi = np.exp(-np.sum((x - X_train[mu:mu + 1].values) ** 2) / (2 * (s ** 2)))
                return phi
            return rbf
        # create Phi
        list_of_rbfs = [create_rbf(m) for m in index_of_mus]
        Phi = list(map(lambda f: np.apply_along_axis(f, 1, dataset), list_of_rbfs))

        # insert bais
        Phi.insert(0, np.repeat(1, np.size(dataset, axis=0)))
        return np.column_stack(Phi)

    def choose_means(num, start=0, end=99):
        arr = []
        tmp = random.randint(start, end)
        for x in range(num):
            while tmp in arr:
                tmp = random.randint(start, end)
            arr.append(tmp)
        arr.sort()
        return arr
    list_of_mus=choose_means(90)

    def calculate_error(dataset_x, dataset_y, l):
        Phi = create_gaussian_functions(X_train, list_of_mus) #100*91
        W = np.linalg.inv((Phi.T @ Phi) + (l * np.identity(91))) @ Phi.T @ Y_train.values  # 91*1
        # prediction
        predicted_y = np.dot(create_gaussian_functions(dataset_x, list_of_mus), W)
        #error
        rmse = np.sqrt(np.sum((predicted_y - dataset_y.values) ** 2) / dataset_y.shape[0])
        return rmse

    #make df of errors
    error_df=pd.DataFrame(columns=['lambda','training_error','test_error'])
    for i in (0,0.01,0.1,10,100,1000):
        error_df.loc[i] = [i] + [calculate_error(X_train,Y_train,i)] +[calculate_error(X_test,Y_test,i)]
    print(error_df)

    plt.semilogx(error_df['lambda'],error_df['training_error'],marker = ".",color='blue',label='trainig error')
    plt.semilogx(error_df['lambda'],error_df['test_error'],marker = ".",color='red',label='test error')
    plt.xlabel('lambda')
    plt.ylabel('RMS error')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    # uncomment the one you want to run, comment the others
    polynomial_regression_all_features()
    # polynomial_regression_3rdfeature()
    # regularized_polynomial_regression_3rdfeature()
    # gaussian_regression_all_features()
    # regularized_gaussian_regression_all_features()
