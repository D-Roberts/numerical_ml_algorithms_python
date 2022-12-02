
import numpy as np 
import copy


def mean(x):
    return sum(x) / len(x)

def get_col_means(X):
    return [mean(x) for x in zip(*X)]

def center_matrix(X):
    Xc = []
    means = get_col_means(X)
    for row in X:
        Xc.append([row[j] - means[j] for j in range(len(X[0]))])
    return Xc

def get_cov_mat(X, y=None):
    if y is None:
        y = X 
    Xc = center_matrix(X)
    return get_dot(get_transpose(Xc), Xc)

def get_transpose(X):
    Xt = [[0] * len(X) for _ in range(len(X[0]))]

    for i in range(len(X)):
        for j in range(len(X[0])):
            Xt[j][i] = X[i][j]
    return Xt

def get_dot(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    res = [[0] * p for _ in range(m)]

    for i in range(m):
        for k in range(p):
            for j in range(n):
                res[i][k] += A[i][j] * B[j][k]
    return res

class PCA:
    def __init__(self, num_comp):
        self.num_comp = num_comp

    def fit(self, X):
        cov_mat = get_cov_mat(X)
        eigval, eigvect = np.linalg.eigh(cov_mat)

        top_inds = sorted(enumerate(eigval), key=lambda x: x[1], reverse=True)[:self.num_comp]
        print(top_inds)
        self.eigval = [eigval[i] for i, _ in top_inds]
       
        self.comp = []
        for row in eigvect:
            self.comp.append([row[i[0]] for i in top_inds])
        return self.comp


    def transform(self, X):
        # trasnpose eigenv and center matrix Vt dot Xc
        return get_dot(get_transpose(self.comp), center_matrix(X))

    def get_var_ratios(self):
        return [x/sum(self.eigval) for x in self.eigval]


def main():
    mypca = PCA(2)
    import sklearn
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    comps = mypca.fit(X)
    projected = mypca.transform(X)
    print(projected)

    var_ratios = mypca.get_var_ratios()
    print(var_ratios)

    from sklearn.decomposition import PCA as skPCA

    skpca = skPCA(2)
    skpca.fit(X)
    print(skpca.explained_variance_ratio_)



main()
