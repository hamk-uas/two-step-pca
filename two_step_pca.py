import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pickle 

def shift_data(data, lag):
    """
    Creates a matrix based on input data and selected lag
    Corresponds to ~X in the paper 
    Arguments:
        data - numpy array, input matrix
        lag - interger, time lag parameter 
    Returns: 
        tildaX - numpy array, ~X from the paper 
    """

    tildaX = data[lag:]
      
    for l in range(1, lag + 1):
        tildaX = np.hstack((tildaX, data[lag - l: -l]))
    return tildaX 

def delta(data, D):
    """
    Calculates matrix dX = X(t) - X(t-D)
    Arguments:
        data - numpy array with input data 
        D - time difference between the samples
    Returns:
        dX - numpy array
    """
    n_samples, n_variables = data.shape
    dX = np.zeros((n_samples - D, n_variables))

    data_shifted = data[D:]
    dX = data_shifted - data[:-D]
    return dX

def calcA(data, lag, D):
    """
    Calculates matrix A, which represents dynamic part of the model
    Arguments:
        data - numpy array with input data
        lag - lag parameter of the model
        D - time difference parameter
    Returns:
        A - numpy array describing dynamic part of the TS-PCA
    """
    X = data
    tildaX = shift_data(X, lag)
    dX = delta(X, D)
    dtildaX = delta(tildaX, D)
    sizeDiff = dX.shape[0] - dtildaX.shape[0]
    dX = dX[sizeDiff:]
    A1 = np.linalg.inv(np.dot(dtildaX.T, dtildaX))
    A2 = np.dot(dtildaX.T, dX)
    A = np.dot(A1,A2)
    return A

def computeT2(T, E):
    """
    Computes T2 metric
    Arguments:
        T - numpy array, score matrix (or principal components, it is the same), usually you pick first "l" PCs, 
            which explain most variance
        E - eigenvalues which correspond to T
    Returns:
        T2 - Hotelling's T2 metric
    """
    T2 = np.zeros((T.shape[0], 1))
    
    for i in range(T.shape[0]):
        for k in range(T.shape[1]):
            T2[i] = (T[i, k] * T[i, k]) / E[k]
    
    return T2

def computeSPE(T):
    """
    Computes SPE metric
    Arguments:
        T - numpy array, score matrix, unlike in Hotelling's T2 you use "m-l" PCs here,
            so T here is all principal components you didnt use in T2 calculations.
    Returns:
        SPE - numpy array, Squared Prediction Error
    """
    SPE = np.zeros((T.shape[0], 1))
    for i in range(T.shape[0]):
        SPE[i] = np.sum(np.square(T[i]))

    return SPE

def pca(data):
    """
    Usual PCA decomposition which returns all intermediate parameters
    Arguments:
        data - numpy array with input data (should be already with zero mean and unit variance)
                most probably you want to pass U(innovation part) here
    Returns:
        T - score matrix (Principal components)
        P - loading matrix (eigenvectors)
        E - eigenvalues
        var_explained - variance explained for each principal component
    """
    cov_matrix = np.cov(data.T)
    eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
    #the next line combines eig_vals and eig_vectors into pairs for further soring
    #numpy.linalg.eig usually returns them sorted, but it is not guaranteed
    eig_pairs = [[np.abs(eig_vals[i]), eig_vectors[:, i]] for i in range(len(eig_vals))]
    print("Eigen_values shape: {}".format(len(eig_pairs)))
    eig_pairs.sort()
    eig_pairs.reverse()
    #computing explained variance for each principal component
    var_explained = [(i/sum(eig_vals)) for i in sorted(eig_vals, reverse=True)]
    #extracting eigen vectors and eigen values from the pairs and converting them to numpy arrays,
    #because I like them better
    P = np.array([i[1] for i in eig_pairs])
    T = np.dot(data, P.T)
    E = np.array([i[0] for i in eig_pairs])

    return T, P, E, var_explained

def dividePCs(T, E, var_explained, var_explained_required):
    """
    Divides Principal component matrix into two matricies, depending on how much variance,
    must be explained. You can think of this function the same way, as n_pca (or n_components) in sklearn.decomposition.PCA
    Arguments:
        T - numpy array, score matrix
        E - numpy array, eigenvalues
        var_explained - list, consists of values, which show how much variance each PC explains
        var_explained_required - float, how much variance must be explained by all selected PCs
    Returns:
        T_l - numpy array, score matrix with top l PCs, which explain required amount of variance
        E_l - numpy array, eigen values which correspond to T_l score matrix
        T_rest - numpy array, score matrix containing all the PCs not included in T_l, it is needed for SPE calculations
    """
    total_var_explained = 0
    n_pca = 0
    for i in var_explained:
        total_var_explained = total_var_explained + i
        n_pca = n_pca + 1
        if total_var_explained >= var_explained_required:
            break

    return T[:, :n_pca], E[:n_pca], T[:, n_pca:]


class TS_PCA:
    scaler = StandardScaler()

    def fit(self, data, qMAX, DMAX, verbose=False):
        """
        Fit TS-PCA model with training data also calculates optimal q and D for the model
        Arguments:
            data - pandas dataframe (or numpy array), training data 
            qMAX - maximum lag parameter value (this value is used for search limit) 
            DMAX - maximum time deference parameter value (this value is used for search limit)
            verbose - boolean, set to true if you want to see how the method proceeds
        """
        #check if data is dataframe, if that is the case, convert it to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        #this part is for q and D selection
        min_var = []
        for i in range(1, qMAX):
            tildaX = shift_data(data, i)
            sizeDiff = data.shape[0] - tildaX.shape[0]
            temp_data = data[sizeDiff:]
            A = calcA(temp_data, i, 100)
            U = temp_data - np.dot(tildaX, A)
            U_v = np.var(U)
            min_var.append(U_v)
            if verbose:
                print("Iteration - q: {}, Variance: {}".format(i, U_v))
        self.q = min_var.index(min(min_var)) + 1
        #and now the same for D
        min_var = []
        tildaX = shift_data(data, self.q)
        sizeDiff = data.shape[0] - tildaX.shape[0]
        data = data[sizeDiff:]
        for i in range(1, DMAX):
            A = calcA(data, self.q, i)
            U = data - np.dot(tildaX, A)
            U_v = np.var(U)
            min_var.append(U_v)
            if verbose:
                print("Iteration - D: {}, Variance: {}".format(i, U_v))
        self.D = min_var.index(min(min_var)) + 1
        print("Optimal parameters are Q: {}, D: {}".format(self.q, self.D))

      
        self.A = calcA(data, self.q, self.D)
        U = data - np.dot(tildaX, self.A)
        self.scaler.fit(U)
        with open("scaler", "wb") as f:
            pickle.dump(self.scaler, f, pickle.HIGHEST_PROTOCOL)
        

    def detect(self, data, var_explained):
        """
        Uses pretrained TS-PCA model to obtain Hotelling's T2 and SPE metrics for the test data
        Arguments:
            data - pandas dataframe or numpy array, testing data 
            var_explained - float (0-1), determines how many principal components are used for T2 and SPE calculations
                            rule of the thumb, keep it around 0.8-0.9
        Returns:
            metrics_df - pandas dataframe with T2 and SPE metrics 
        """
        #check if input data is a dataframe and convert in to numpy array if needed 
        if isinstance(data, pd.DataFrame):
            data = data.values 
        
        tildaX = shift_data(data , self.q)
        sizeDiff = data.shape[0] - tildaX.shape[0] #it should be equal to "q", but this check is fast
        #shift_data removes first q elements, so reallignment is required 
        data = data[sizeDiff:]
        #compute innovation part 
        U = data - np.dot(tildaX, self.A)
        #zero mean unit variance normalizing
        with open("scaler", "rb") as f:
            self.scaler = pickle.load(f)
            
        U = self.scaler.transform(U)
        T, P, E, var_explained_list = pca(U)
        T_l, E_l, T_rest = dividePCs(T, E, var_explained_list, var_explained)
        T2 = computeT2(T_l, E_l)
        SPE = computeSPE(T_rest)
        metrics_df = pd.DataFrame(data=np.concatenate((T2, SPE), axis=1), columns=['T2', 'SPE'])

        return metrics_df