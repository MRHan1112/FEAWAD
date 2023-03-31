import argparse
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import os

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def dataLoading(dataset):
  x = []
  labels = []
  dim = 0
  path = 'dataset/' + dataset + '.csv'
  with open(path, 'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
      if dataset == 'spambase_normalization':
        dim = len(i) - 1
      else:
        dim = len(i) - 2
      x.append(i[0:dim])
      labels.append(i[dim])
    for m in range(len(x)):
      for n in range(dim):
        x[m][n] = float(x[m][n])
    for k in range(len(labels)):
      labels[k] = float(labels[k])
  x = np.array(x, dtype=np.float32)
  labels = np.array(labels, dtype=np.float32)
  return x, labels, dim


def writeResults(dataset, mean_aucroc, mean_aucpr, std_aucroc, std_aucpr):
  path = "result.csv"
  with open(path, 'a') as f:
    row = "dataset is " + dataset + ", mean_aucroc is " + str(mean_aucroc) + ", mean_aucpr is " + str(mean_aucpr) + ", std_aucroc is " + str(std_aucroc) + ", std_aucpr is " + str(std_aucpr) + '\n'
    f.write(row)
  
  
def injectNoise(X_train, outlier_indices, n_cont, rand_seed):
  exchange_ratio = 0.05
  outlier_sample = X_train[outlier_indices]
  size, features = outlier_sample.shape
  noises_sample = np.empty((n_cont, features))
  n_exchange_features = int(features * exchange_ratio)
  rng = np.random.RandomState(rand_seed)
  outlier1, outlier2 = rng.choice(outlier_indices, 2, replace=False)
  exchange_features = rng.choice(features, n_exchange_features, replace=False)
  for i in range(n_cont):
    noises_sample[i] = X_train[outlier1].copy()
    noises_sample[i, exchange_features] = X_train[outlier2][exchange_features]
  return noises_sample


class AEModel(torch.nn.Module):
  def __init__(self, dim) -> None:
    super().__init__()
    self.dim = dim
    self.en1 = torch.nn.Linear(dim, 128)
    self.en2 = torch.nn.Linear(128, 64)
    self.de1 = torch.nn.Linear(64, 128)
    self.de2 = torch.nn.Linear(128, dim)
    self.net = torch.nn.ModuleList([self.en1, self.en2, self.de1, self.de2])
  def forward(self, x):
    x = x.type(torch.float32)
    xen1 = torch.nn.functional.relu(self.en1(x))
    xen2 = torch.nn.functional.relu(self.en2(xen1))
    xde1 = torch.nn.functional.relu(self.de1(xen2))
    xde2 = torch.nn.functional.relu(self.de2(xde1))
    return xen2, xde2

def aesampler(X_train, batch_size, rng):
  ref = np.empty(batch_size, dtype=int)
  n_train = X_train.shape[0]
  for i in range(batch_size):
    inlier_id = rng.choice(n_train, 1)
    ref[i] = inlier_id
  return np.array(X_train[ref], dtype=np.float32)

def aefit(X_train, aemodel, epoch, batch_size, rng, device):
  lossfunc = torch.nn.MSELoss(reduction='mean')
  lossfunc = lossfunc.to(device)
  optimizer = torch.optim.Adam(params=aemodel.net.parameters(), lr=0.0001)
  for i in range(epoch):
    for j in range(0, 20):
      optimizer.zero_grad()
      data = aesampler(X_train, batch_size, rng)
      data = torch.from_numpy(data).to(device)
      en2, de2 = aemodel(data)
      en2 = en2.to(device)
      de2 = de2.to(device)
      loss = lossfunc(data.float(), de2.float())    
      loss.backward()
      optimizer.step()
    print("ae epoch is {}, loss is {}".format(i + 1, loss))

class DevModel(torch.nn.Module):
  def __init__(self, aemodel, dim):
    super().__init__()
    self.aemodel = aemodel
    self.fc1 = torch.nn.Linear(dim + 64 + 1, 256)
    self.fc2 = torch.nn.Linear(256 + 1, 32)
    self.fc3 = torch.nn.Linear(32 + 1, 1)
    self.net = torch.nn.ModuleList([self.aemodel, self.fc1, self.fc2, self.fc3])
  def forward(self, x):
    en2, de2 = self.aemodel(x)
    sub_result = torch.sub(de2, x)
    recon_error = torch.norm(sub_result, p=2, dim=1)
    recon_error = recon_error.reshape(-1, 1)
    residual_error = sub_result / (recon_error + 1e-5)

    input_tensor = torch.cat([recon_error, residual_error, en2], dim=1)
    intermediate = torch.nn.functional.relu(self.fc1(input_tensor))
    intermediate = torch.cat([intermediate, recon_error], dim=1)
    intermediate = torch.nn.functional.relu(self.fc2(intermediate))
    intermediate = torch.cat([intermediate, recon_error], dim=1)
    intermediate = self.fc3(intermediate)
    return intermediate


def dev_loss(label, recon_error, pred):
  loss1 = torch.mean((1 - label) * recon_error + label * (torch.max(5 - recon_error, torch.zeros_like(recon_error))))
  loss2 = torch.mean((1 - label) * torch.squeeze(torch.abs(pred), dim=-1) + label * torch.squeeze(torch.max(5 - pred, torch.zeros_like(pred)), dim=-1))
  loss = loss1 + loss2
  return loss


def devsampler(X_train, y_train, batch_size, rng):
  ref = np.empty(batch_size, dtype=int)
  label = np.empty(batch_size)
  outlier_indices = np.where(y_train == 1)[0]
  inlier_indices = np.where(y_train == 0)[0]
  for i in range(batch_size):
    if(i % 2 == 0):
      outlier_id = rng.choice(outlier_indices, 1)
      ref[i] = outlier_id
      label[i] = y_train[outlier_id]
    else:
      inlier_id = rng.choice(inlier_indices, 1)
      ref[i] = inlier_id
      label[i] = y_train[inlier_id]
  return np.array(X_train[ref], dtype=np.float32), np.array(label, dtype=np.float32)
    

def devfit(X_train, y_train, devmodel, aemodel, epoch, batch_size, rng, device):
  optimizer = torch.optim.Adam(params=devmodel.net.parameters(), lr=0.0001)
  for i in range(epoch):
    for j in range(0, 20):
      optimizer.zero_grad()
      data, label = devsampler(X_train, y_train, batch_size, rng)  
      data = torch.from_numpy(data).to(device)
      label = torch.from_numpy(label).to(device)
      en2, de2 = aemodel(data)
      de2 = de2.to(device)
      sub_result = torch.sub(de2, data)
      recon_error = torch.norm(sub_result, p=2, dim=1)
      pred = devmodel(data)
      muti_loss = dev_loss(label, recon_error, pred)
      muti_loss.backward()
      optimizer.step()
    print("dev epoch is {}, loss is {}".format(i + 1, muti_loss))



def run(args):
  datasets = [os.path.splitext(_)[0] for _ in os.listdir(path='dataset')
              if os.path.splitext(_)[1] == '.csv']
  rand_seed = args.rand_seed
  runs = args.runs
  known_outliers = args.known_outliers
  cont_rate = args.cont_rate
  epoch = args.epoch
  batch_size = args.batch_size
  device = torch.device("cuda:0")
  for dataset in datasets:
    dataset = dataset.strip()
    aucroc = torch.zeros(runs)
    aucpr = torch.zeros(runs)
    x, labels, dim = dataLoading(dataset)
    for i in range(runs):
      print("{} :round is {}".format(dataset, i + 1))
      test_ratio = 0.2
      X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=test_ratio, random_state=rand_seed, stratify=labels)
      outlier_indices = np.where(y_train == 1)[0]
      inlier_indices = np.where(y_train == 0)[0]
      n_inlier = inlier_indices.shape[0]
      n_outlier = outlier_indices.shape[0]
      rng = np.random.RandomState(rand_seed)
      print("Original training size is {}, Number of outliers in Training data is {}".format(X_train.shape[0], n_outlier))
      
      # Process labeled data
      if n_outlier > known_outliers:
        n_delete = n_outlier - known_outliers
        delete_indices = rng.choice(outlier_indices, size=n_delete, replace=False)
        X_train = np.delete(X_train, delete_indices, axis=0)
        y_train = np.delete(y_train, delete_indices, axis=0)
        n_outlier = known_outliers
        
      # Inject noises
      outlier_indices = np.where(y_train == 1)[0]
      n_cont = int(cont_rate * n_inlier / (1 - cont_rate))
      cont_sample = injectNoise(X_train, outlier_indices, n_cont, rand_seed)
      X_train = np.append(X_train, cont_sample, axis=0)
      y_train = np.append(y_train, np.zeros(n_cont), axis=0)
      print("Processed Train data number is {}, outliers number in Training data is {}, normal number in Traning data is {}".format(X_train.shape[0], n_outlier, n_inlier))
      n_outlier_test = np.where(y_test == 1)[0].shape[0]
      n_inlier_test = np.where(y_test == 0)[0].shape[0]
      print("Test data number is {}, outliers number is test data is {}, normal number in test data is {}".format(X_test.shape[0], n_outlier_test, n_inlier_test))

      #Pretrain AEModel
      print("pre-training start...")
      aemodel = AEModel(dim)
      aemodel = aemodel.to(device)
      aefit(X_train, aemodel, epoch, batch_size, rng, device)
      print("load autoencoder model...")
      
      #Train system
      print("end-to-end training start...")
      dev_model = DevModel(aemodel, dim)
      dev_model = dev_model.to(device)
      devfit(X_train, y_train, dev_model, aemodel, epoch, batch_size, rng, device)
      print("load model and print results...")
      
      X_test = torch.from_numpy(X_test).to(device)
      pred = dev_model(X_test).squeeze(1)
      pred = pred.cpu().detach().numpy()
      aucroc[i] = roc_auc_score(y_score=pred, y_true=y_test)
      aucpr[i] = average_precision_score(y_score=pred, y_true=y_test)
      print("run is {}, aucroc is {:.6f}".format(i + 1, aucroc[i]))
      print("run is {}, aucpr is {:6f}".format(i + 1, aucpr[i]))
      
    mean_aucroc = torch.mean(aucroc).item()
    std_aucroc = torch.std(aucroc).item()
    mean_aucpr = torch.mean(aucpr).item()
    std_aucpr = torch.std(aucpr).item()
    print("average AUC-ROC is {:.6f}, average AUC-PR is {:.6f}".format(mean_aucroc, mean_aucpr))
    print("std AUC-ROC is {:.6f}, std AUC-PR is {:.6f}".format(std_aucroc, std_aucpr))
    writeResults(dataset, mean_aucroc, mean_aucpr, std_aucroc, std_aucpr)


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiment to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--epoch", type=int, default=30, help="the number of epochs")
parser.add_argument("--batch_size", type=int, default=512, help="the size of the batch")
parser.add_argument("--rand_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
run(args)




