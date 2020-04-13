import numpy as np

class ProcessPCA:

    def __init__(self, n_component):
        
        self.n_component = n_component

    def fit(self, data):

        self.mean = data.mean(axis=0, keepdims=True)

        mean_adj = data - self.mean

        if data.shape[0] > data.shape[1]:
            cov = np.matmul(mean_adj.T, mean_adj)/(data.shape[0] - 1)       
            self.eigen_vect, self.eigen_value, _ = np.linalg.svd(cov)

        else:
            cov = np.matmul(mean_adj, mean_adj.T)/(data.shape[0] - 1)
            self.eigen_vect, self.eigen_value, _ = np.linalg.svd(cov)
            self.eigen_vect = np.matmul(data.T, self.eigen_vect[:, :data.shape[1]])
            self.eigen_vect /= np.linalg.norm(self.eigen_vect, axis=0)

        self.calc_cr_and_ccr()
        self.dim_reduce = self.eigen_vect[:, :self.n_component]
    
    def get_cr(self):
        return self.c_rate
    
    def get_ccr(self):
        return self.cc_rate

    def set_mean_vect(self, mean_vect):
        self.mean = mean_vect

    def get_mean_vect(self):
        return self.mean
    
    def set_eigen_value(self, eigen_value):
        self.eigen_value = eigen_value
        
    def get_eigen_value(self):
        return self.eigen_value
    
    def set_eigen_vect(self, eigen_vect):
        self.eigen_vect = eigen_vect
        self.dim_reduce = self.eigen_vect[:, :self.n_component]

        
    def get_eigen_vect(self):
        return self.eigen_vect

    def projection(self, x):
        mean_adj = x - self.mean
        project = np.matmul(mean_adj, self.dim_reduce)
        
        return project
    
    def reconstruction(self, z):
        data_rec = np.matmul(z, self.dim_reduce.T) + self.mean
        
        return data_rec
    
    def calc_cr_and_ccr(self):

        total_eigen = np.sum(self.eigen_value)
        
        self.c_rate = list(self.eigen_value/total_eigen)

        total = 0
        self.cc_rate = []
        for cr in self.c_rate:
            total += cr
            self.cc_rate.append(total)
        


def debug():

    x = np.array([[1, 0], [0, 1], [1, 1]])

    pca = ProcessPCA(2)
    pca.fit(x)

    print("--------------------------------------------------")
    print("#### Debug #######################################")
    print("--------------------------------------------------\n")
    print("mean_vector={}".format(pca.get_mean_vect()))
    print("eigen_vector={}".format(pca.get_eigen_vect()))
    print("eigen_value={}\n".format(pca.get_eigen_value()))
    print("--------------------------------------------------")
    print("##################################################")
    print("--------------------------------------------------")



if __name__ == "__main__":
    debug()