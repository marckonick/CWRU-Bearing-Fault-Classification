import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
"""
    Model definisions and evaluation functions 
"""
class labeled_dataset(Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])
      
        
class DNN_MEL(nn.Module):

    def __init__(self, in_shape, n_classes, n_per_layer = [64,32]):
        super().__init__()

        self.lin1_e = nn.Linear(in_shape, n_per_layer[0])  # add stride=(1,2) to each layer
        self.lin2_e = nn.Linear(n_per_layer[0], n_per_layer[1])
        self.lin3_e = nn.Linear(n_per_layer[1], n_classes)

        self.bn1 = nn.BatchNorm1d(n_per_layer[0])
        self.bn2 = nn.BatchNorm1d(n_per_layer[1])

    def forward(self, x):
        
        out = self.lin1_e(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.lin2_e(out)
        out = self.bn2(out)
        out = torch.relu(out)


        out = self.lin3_e(out)

        return out

    def number_of_params(self):
         print('Number of network paramaters:')
         print(sum(p.numel() for p in self.parameters()))



class VGG_1D(nn.Module):
    def __init__(self, n_classes, in_channels = 2, n_chans1=[8,16,16, 16], k_size = [3,3,3,3], padding_t='same', fc_size = 960):
        super().__init__()
        self.chans1=n_chans1

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.chans1[0], kernel_size=(3,k_size[0]),  padding=padding_t)  # add stride=(1,2) to each layer
        self.conv2 = nn.Conv2d(in_channels=n_chans1[0], out_channels=self.chans1[1], kernel_size=(3,k_size[0]), padding=padding_t)

        self.conv3 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[1], kernel_size=(3,k_size[1]),  padding=padding_t)
        self.conv4 = nn.Conv2d(in_channels=self.chans1[1],out_channels=self.chans1[2], kernel_size=(3,k_size[1]), padding=padding_t)

        self.conv5 = nn.Conv2d(in_channels=self.chans1[2],out_channels=self.chans1[2], kernel_size=(3,k_size[2]),  padding=padding_t)
        self.conv6 = nn.Conv2d(in_channels=self.chans1[2],out_channels=self.chans1[3], kernel_size=(3,k_size[2]),   padding=padding_t)

        #self.gap = GAP1d(1)
        self.fc1 = nn.Linear(fc_size, n_classes)

    def forward(self, x):
        #print(x.size())
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = F.max_pool2d(out, stride=2, kernel_size=2)
        #print(out.size())

        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = F.max_pool2d(input=out, stride=2, kernel_size=2)
        #print(out.size())

        out = torch.relu(self.conv5(out))
        out = torch.relu(self.conv6(out))
        out = F.max_pool2d(input=out, stride=2, kernel_size=2)
        #print(out.size())


        #out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2) # N C H W
        out = out.reshape(out.shape[0], -1)
        #out = self.gap(out)
        #print(out.size())
        out = self.fc1(out)


        return out

    def number_of_params(self):
         print('Numer of network paramteres:')
         print(sum(p.numel() for p in self.parameters()))



"""
   Test Function
"""

def test_model(X_test, model, n_classes, Y_all_names, device='cpu'):
    
    
 preds_all = []
 y_all = []

 model.eval()
 with torch.no_grad():
    for x, y in X_test:
         x = x.to(device=device)
         y = y.to(device=device)

         outputs = model(x.float())    
         outputs = torch.softmax(outputs, dim=1)
         _, predicted = torch.max(outputs.data, 1)
         preds_all.append(predicted.cpu().numpy())
         y_all.append(y.cpu().numpy())

 preds_all = np.concatenate(preds_all)
 y_all = np.concatenate(y_all)

 acc = sum(preds_all == y_all)/len(preds_all)
 print(f"Accuracy: {100*acc}")
 
 cm = confusion_matrix(y_all, preds_all)
 
 cm_norm = copy.copy(cm)
 cm_norm = np.array(cm_norm, dtype=float)
 
 for i in range(0, n_classes):
   cm_norm[i,:] = cm_norm[i,:]/np.sum(cm, axis=1)[i]

 print('\nConfusion matrix: ' )
 print(np.array_str(cm_norm, precision=4, suppress_small=True))
 print('\n')
 

 
 return cm, cm_norm

    
#sns.heatmap(cm_norm, annot=True, cmap='Blues', cbar=False, xticklabels=Y_names_list, yticklabels=Y_names_list)
#figure = sv.get_figure()    
#figure.savefig('Results/svm_conf.png', dpi=400)

#figure = sv.get_figure()    
#figure.savefig('Results/svm_conf.png', dpi=400)

