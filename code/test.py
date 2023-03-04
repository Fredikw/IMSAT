"""
Libraries

"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics.cluster import normalized_mutual_info_score
from munkres import Munkres, print_matrix


from torch.utils.data import Dataset, DataLoader



"""
Settings
"""

lr = 0.002
batch_size = 250
lam = 0.1
mu = 4
prop_eps = 0.25
hidden_list = '1200-1200'
n_epoch = 10#50
dataset = "mnist"





# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




"""
Data Preprocessing

"""
#TODO Preprocess the AILARON dataset to a suitable format.
#TODO Implement custome dataset for AILARON data. Should inherit from torch.utils.data.Dataset

import torch
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, train, download):
        self.mnist_data = torchvision.datasets.MNIST(root='./data',
                                                     train=train,
                                                     download=True,
                                                     transform=torchvision.transforms.ToTensor())

    def __getitem__(self, idx):
        img, label = self.mnist_data[idx]
        return img, label, idx

    def __len__(self):
        return len(self.mnist_data)

# ailaron_train = AILARONDataset()
# dataloader = DataLoader(dataset=ailaron_train, batch_size=batch_size, shuffle=True)

# Load MNIST dataset, normalizes data and transform to tensor.
mnist_train = MNISTDataset(train=True, download=True)
mnist_test  = MNISTDataset(train=False, download=False)

# Create a subset of the MNIST dataset with the first 100 examples
mnist_train_subset = torch.utils.data.Subset(mnist_train, range(3000))
mnist_test_subset  = torch.utils.data.Subset(mnist_test, range(32))

# # Get a random image from the dataset
# image, label = mnist_train[np.random.randint(0, len(mnist_train))]

# # Plot the image
# plt.imshow(image[0], cmap='gray')
# plt.title(f'Label: {label}')
# plt.show()

# Create DataLoader
train_loader = DataLoader(mnist_train_subset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)









tot_cl = 10

p_pred = np.zeros((len(mnist_train),10))
y_pred = np.zeros(len(mnist_train))
y_t = np.zeros(len(mnist_train))

# Deep Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The input layer takes in 28 x 28 = 784 dimensional data, which is then transformed into a hidden fully connected (fc) layer with 1200 neurons.
        self.fc1 = nn.Linear(28 * 28, 1200)
        # initializes the weights of 'fc1' using a normal distribution with a standard deviation of '0.1 * math.sqrt(2/(28*28))' - heuristic that is often used in practice to initialize the weights of fully connected layers
        torch.nn.init.normal_(self.fc1.weight,std=0.1*math.sqrt(2/(28*28)))
        # Initialize bias term to 0.
        self.fc1.bias.data.fill_(0)
        
        self.fc2 = nn.Linear(1200, 1200)
        torch.nn.init.normal_(self.fc2.weight,std=0.1*math.sqrt(2/1200))
        self.fc2.bias.data.fill_(0)
        
        self.fc3 = nn.Linear(1200, 10)
        torch.nn.init.normal_(self.fc3.weight,std=0.0001*math.sqrt(2/1200))
        self.fc3.bias.data.fill_(0)
        
        # Batch normalization (bn)
        self.bn1  = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F= nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.bn2  = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F= nn.BatchNorm1d(1200, eps=2e-5, affine=False)
    
    def forward(self, x, update_batch_stats=True):
        if not update_batch_stats:
            x = self.fc1(x)
            x = self.bn1_F(x)*self.bn1.weight+self.bn1.bias
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2_F(x)*self.bn2.weight+self.bn2.bias
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

net = Net()
if use_cuda:
    net.to(device)

# Loss function and optimizer
# Compute entropy according to the entropy entropy function
def entropy(p):
    # compute entropy
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-8)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-8))
    else:
        raise NotImplementedError

def Compute_entropy(net, x):

    # Mariginal probability
    p = F.softmax(net(x),dim=1)

    # Conditional probability
    p_ave = torch.sum(p, dim=0) / len(x)
    
    # Compute marginal entropy and conditional entropy respectively
    return entropy(p), entropy(p_ave)


'''
Code for R_sat
'''

# From equation 10
def kl(p, q):
    # compute KL divergence between p and q
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))
def distance(y0, y1):
    # compute KL divergence between the outputs of the newtrok
    return kl(F.softmax(y0,dim=1), F.softmax(y1,dim=1))


def vat(network, distance, x, eps_list, xi=10, Ip=1):
    # compute the regularized penality [eq. (4) & eq. (6), 1]
    
    # Compute y
    with torch.no_grad():
        y = network(Variable(x))

    # Comput r from random distibution U, i guess
    d = torch.randn((x.size()[0],x.size()[1]))

    # Normalize so the length of d is 1.
    d = F.normalize(d, p=2, dim=1)

    # # TODO remove
    # print("d.grad", d.grad)
    # print("d.grad_fn", d.grad_fn)
    # print("d.is_leaf", d.is_leaf)
    # print("d.requires_grad", d.requires_grad)

    # exit()


    # Is this a sequence of forward and backward passes to find the correct pertubation
    for ip in range(Ip):
        
        # Variable is a wrapper, neccesary in order to compute the gradient, do backpropagation etc.
        d_var = Variable(d)
        if use_cuda:
            d_var = d_var.to(device)
        d_var.requires_grad_(True)

        # Applying the pertubation T(x) = x + d (which is r in the literature)
        y_p = network(x + xi * d_var)

        # Computing the kl_loss = Kullback-Leibler divergence
        # tensor(-3.4246e-09, grad_fn=<DivBackward0>) grad_fn specifies the gradient function associated with the tensor.
        kl_loss = distance(y,y_p)

        # Triggers backpropagation, i.e. computes the gradient 
        kl_loss.backward(retain_graph=True)

        # The backpropagation method above sets the .grad attribute of all variables that are included in the computation of kl_loss
        d = d_var.grad

        # Normalize so that the length of d is 1 along each row(dim=1)
        d = F.normalize(d, p=2, dim=1)
    
    #Updating d to be the last d from the itterations above
    d_var = d
    
    if use_cuda:
        d_var = d_var.to(device)

    
    eps = prop_eps * eps_list
    eps = eps.view(-1,1)
    y_2 = network(x + eps*d_var)

    return distance(y,y_2)

"""
What does this do?????????

"""
def enc_aux_noubs(x):
    # not updating gamma and beta in batchs
    return net(x, update_batch_stats=False)

def loss_unlabeled(x, eps_list):
    # to use enc_aux_noubs
    L = vat(enc_aux_noubs, distance, x, eps_list)
    return L

'''

'''
def upload_nearest_dist(dataset):
    # Import the range of local perturbation for VAT
    nearest_dist = np.loadtxt('10th_neighbor.txt').astype(np.float32)
    return nearest_dist

optimizer = optim.Adam(net.parameters(), lr=lr)


'''
performance metric
'''
def compute_accuracy(y_pred, y_t):
    # compute the accuracy using Hungarian algorithm
    m = Munkres()
    mat = np.zeros((tot_cl, tot_cl))
    for i in range(tot_cl):
        for j in range(tot_cl):
            mat[i][j] = np.sum(np.logical_and(y_pred == i, y_t == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(tot_cl):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in y_pred]
    acc = np.sum(pred_corresp == y_t) / float(len(y_t))
    return acc



'''
Training
'''

print('==> Start training..')
nearest_dist = torch.from_numpy(upload_nearest_dist(dataset))
if use_cuda:
    nearest_dist = nearest_dist.to(device)
best_acc = 0
for epoch in range(n_epoch):
    net.train()
    running_loss = 0.0
    #   start_time = time.clock()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels, ind = data
        inputs = inputs.view(-1, 28 * 28)
        if use_cuda:
            inputs, labels, nearest_dist, ind = inputs.to(device), labels.to(device), nearest_dist.to(device), ind.to(device)
        
        # forward
        aver_entropy, entropy_aver = Compute_entropy(net, Variable(inputs))
        
        # Representing mutual information as the difference between marginal entropy and conditional entropy
        r_mutual_i = aver_entropy - mu * entropy_aver
        
        # Regularization penalty, R_sat
        loss_ul = loss_unlabeled(Variable(inputs), nearest_dist[ind])
        
        # Regularized Information Maximization (RIM) objective.
        loss = loss_ul + lam * r_mutual_i
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # loss accumulation
        running_loss += loss.item()

    """
    wtf is this????????
    """

    # statistics
    net.eval()
    p_pred = np.zeros((len(mnist_train),10))
    y_pred = np.zeros(len(mnist_train))
    y_t = np.zeros(len(mnist_train))

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels, ind = data
            inputs = inputs.view(-1, 28 * 28)
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs=F.softmax(net(inputs),dim=1)

            """
            Wtf is happening here!?!?!?!
            """

            y_pred[i*batch_size:(i+1)*batch_size]=torch.argmax(outputs,dim=1).cpu().numpy()
            p_pred[i*batch_size:(i+1)*batch_size,:]=outputs.detach().cpu().numpy()
            y_t[i*batch_size:(i+1)*batch_size]=labels.cpu().numpy()
    

    acc = compute_accuracy(y_pred, y_t)
    print("epoch: ", epoch+1, "\t total lost = {:.4f} " .format(running_loss/(i+1)), "\t MI = {:.4f}" .format(normalized_mutual_info_score(y_t, y_pred)), "\t acc = {:.4f} " .format(acc))

    # save the "best" parameters
    if acc > best_acc:
        best_acc = acc
    # show results

print("Best accuracy = {:.4f}" .format(best_acc))
print('==> Finished Training..')




"""
Evaluation Metric

"""
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

# TODO consider including Normalized Information Score as an evaluation metric.

def unsupervised_clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the unsupervised clustering accuracy between two clusterings.
    Uses the Hungarian algorithm to find the best matching between true and predicted labels.

    Args:
        y_true: true cluster labels as a 1D torch.Tensor
        y_pred: predicted cluster labels as a 1D torch.Tensor

    Returns:
        accuracy: unsupervised clustering accuracy as a float
    """
    # Create confusion matrix
    cm = confusion_matrix(y_pred, y_true)

    # Compute best matching between true and predicted labels using the Hungarian algorithm
    _, col_ind = linear_sum_assignment(-cm)

    # Reassign labels for the predicted clusters
    y_pred_reassigned = torch.tensor(col_ind)[y_pred.long()]

    # Compute accuracy as the percentage of correctly classified samples
    acc = accuracy_score(y_true, y_pred_reassigned)

    return acc

"""
Testing

"""

def test_classifier(model: Net, test_loader: DataLoader) -> None:
    """
    Testing a classifier given the model and a test set.

    Args:
        model: Neural network model to train.
        test_loader: PyTorch data loader containing the test data.
    
    Returns:
        None
    """
    
    # Disable gradient computation, as we don't need it for inference
    model.eval()
    # Initialize tensors for true and predicted labels
    y_true = torch.zeros(len(test_loader.dataset))
    y_pred = torch.empty(len(test_loader.dataset))

    with torch.no_grad():
        # Iterate over the mini-batches in the data loader
        for i, data in enumerate(test_loader):
            # Get the inputs and true labels for the mini-batch and reshape
            inputs, labels_true, _ = data
            inputs = inputs.view(-1, 28*28)

            # Forward pass through the model to get predicted labels
            labels_pred = F.softmax(model(inputs), dim=1)

            # Store predicted and true labels in tensors
            y_pred[i*len(labels_true):(i+1)*len(labels_true)] = torch.argmax(labels_pred, dim=1)
            y_true[i*len(labels_true):(i+1)*len(labels_true)] = labels_true

    # Compute unsupervised clustering accuracy score
    acc = unsupervised_clustering_accuracy(y_true, y_pred)

    print(f"\nThe unsupervised clustering accuracy score of the classifier is: {acc}")


test_classifier(net, train_loader)