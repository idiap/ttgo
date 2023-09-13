
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

class GMM:
    """
        Mixture of spherical Gaussians (un-normalized)
        nmix: number of mixture coefficients
        n: dimension of the domain
        s: variance
        mu: the centers assumed to be in : [-L,L]^n
    """
    def __init__(self, n=2, nmix=3, L=1, mx_coef=None, mu=None, s=0.2, device='cpu'):
        self.device = device
        self.n = n # dim
        self.nmix = nmix # number of components
        self.L = L # boundary

        self.s = s # assuming spherical Gaussians
        self.std = (s*torch.ones(self.n)).view(1,self.n).expand(self.nmix,-1).to(device)
        
        if mu is None:
            self.generate_gmm_params()
        else:
            self.mx_coef = mx_coef.to(self.device)
            self.mu = mu.to(self.device)
        self.mv_normals = [torch.distributions.MultivariateNormal(self.mu[k], (self.s**2)*torch.eye(self.n).to(device)) for k in range(self.nmix)]

    def generate_gmm_params(self):
        self.mx_coef = torch.rand(self.nmix).to(self.device) 
        self.mx_coef = self.mx_coef/torch.sum(self.mx_coef)
        self.mu = (torch.rand(self.nmix,self.n).to(self.device)-0.5)*2*self.L

    def pdf(self, x):
        prob = torch.tensor([0]).to(self.device)
        for k in range(self.nmix):
            pdf_k = torch.exp(self.mv_normals[k].log_prob(x))
            prob = prob + self.mx_coef[k]*pdf_k  #*normalize_k*torch.exp(-0.5*l.view(-1)/self.s)
        return prob
    
    def log_pdf(self,x):
        return torch.log(1e-6+self.pdf(x))
    
    def generate_sample(self, n_samples):
        X_noise = torch.randn(n_samples, self.n).to(self.device)
        idx_comp = torch.multinomial(self.mx_coef.view(-1), n_samples, replacement=True).view(-1)
        X = self.mu[idx_comp] + self.s*X_noise
        return X


class BGMM:
    def __init__(self, nmix):
        self.nmix = nmix
        self.model = BayesianGaussianMixture(n_components=nmix, covariance_type='full')
    
    def load_data(self, data):
        # data should be a numpy array: n_samples x dim
        self.data = data
    
    def fit(self):
        self.model.fit(self.data)

    def pdf(self, x):    
        x = torch.from_numpy(x)
        prob = torch.tensor([0])
        for k in range(self.nmix):
            mu = torch.from_numpy(self.model.means_[k]).view(-1)
            cov = torch.from_numpy(self.model.covariances_[k])
            mv_normal = torch.distributions.MultivariateNormal(mu, cov)
            pdf_k = torch.exp(mv_normal.log_prob(x))
            prob = prob + self.model.weights_[k]*pdf_k  
        return prob
        
    # def pdf(self, X):
    #     return np.exp(self.model.score_samples(X))
    
    def log_pdf(self, x):
        return np.log(1e-6+self.pdf(x))

class NNModel(nn.Module):
    def __init__(self, dim, width=64):
        super(NNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y
    

class NeuralNetwork(nn.Module):
    def __init__(self, dim, width=64, lr=1e-3, device='cpu'):
        super(NeuralNetwork, self).__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.model = NNModel(dim, width).to(device)
        self.loss_fcn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def load_data(self, train_data, test_data):
        self.train_data = train_data.to(self.device)
        self.test_data = test_data.to(self.device)

    def train(self, num_epochs=10, batch_size=128, verbose=False):
        size = self.train_data.shape[0]
        for k in range(num_epochs):
            counter = 0
            loss_train = 0.
            counter_batch = 0
            for i in range(int(size/batch_size)-1):
                # Compute prediction and loss
                next_counter = (counter+batch_size)
                x_data = self.train_data[counter:next_counter,:-1]
                y_data = self.train_data[counter:next_counter,-1].view(-1,1)
                y_pred = self.model(x_data)
                loss = self.loss_fcn(y_pred, y_data)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                counter = 1*next_counter
                loss_train += loss.item()
                counter_batch += 1
            loss_train = loss_train/counter_batch
            loss_test = self.test()
            if verbose:
                print(f"epoch:{k}, loss-train:{loss_train}, loss-test:{loss_test}")
            
    def test(self):
        self.model.eval()
        x_data = self.test_data[:,:-1]
        y_data = self.test_data[:,-1].view(-1,1)
        with torch.no_grad():
            pred = self.model(x_data)
            test_loss = self.loss_fcn(pred, y_data).item()
        return test_loss
    
        

# def FitNN(x_train, y_train, x_test, y_test, 
#           learning_rate = 1e-3,batch_size = 128, epochs=10, device="cpu"):
#     dim = x_train.shape[-1]
#     data_train = torch.cat((x_train.view(-1,dim),y_train.view(-1,1)),dim=-1)
#     data_test = torch.cat((x_test.view(-1,dim),y_test.view(-1,1)),dim=-1)
#     model = NeuralNetwork(dim=x_train.shape[-1],width=128).to(device)#width=dim*nmix*10
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     t1 = time.time()
#     for t in range(epochs):
# #         print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(data_train, model, loss_fn, optimizer, batch_size)
#         model.eval()
#         test_loop(data_test, model, loss_fn)
#     t2 = time.time()
#     y_nn_0 = model(x_test)
#     mse_nn_0 = (((y_nn_0.view(-1)-y_test.view(-1))/(1e-9+y_test.view(-1).abs()))**2).mean()
#     return (mse_nn_0, (t2-t1))