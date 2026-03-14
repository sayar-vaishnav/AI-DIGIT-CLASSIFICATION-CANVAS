import torch 
from torch.utils.data import Dataset , DataLoader , random_split 
import torch.nn as nn 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

df = '/Users/sayar/Desktop/canvas/MNSIT CANVAS/msnit_data.csv'

class MNSITDATA:
    def __init__(self, csv_file):
        
        df = pd.read_csv(csv_file)

        y = df.iloc[:, 0].to_numpy()  # label column
        x = df.iloc[:, 1:].to_numpy().astype('float32') / 255.0  # pixels

        self.X = torch.from_numpy(x)
        self.Y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] 
    
full_dataset = MNSITDATA(df)  

class MNSITPYTORCH: 
    
    def __init__(self,csv_path,lr=0.05,epochs=30,batch_size=64): 
        
        self.csv_path = csv_path 
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size 

        self.train_losses = []
        self.test_accuracies = []

        full_dataset = MNSITDATA(csv_path) 

        test_size = 0.2 
        dataset_size = len(full_dataset)
        test_len = int(test_size*dataset_size)
        train_len = dataset_size - test_len

        generator = torch.Generator().manual_seed(42)

        train_dataset , test_dataset = random_split(full_dataset , [train_len,test_len] , generator=generator)

        train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False) 

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = "mps" if torch.backends.mps.is_available() else "cpu" 
        
        self.model = nn.Sequential(
            
            nn.Linear(784,256), # hidden_layer_1 , 256 neurons
            nn.ReLU(), 

            nn.Linear(256,128), # hidden_layer_2 , 128 neurons
            nn.ReLU(), 

            nn.Linear(128,10)   
        ).to(self.device)
        
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)


    def looping(self): 
        for epochs in range(self.epochs): 
            train_loss = 0.0
            for x_batch , y_batch in self.train_loader: 
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(x_batch)
                loss = self.criterion(outputs,y_batch)
                self.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step() 
                train_loss += loss.item() 
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)    
            correct = 0 
            total = 0 
            with torch.no_grad(): 
                for xt_b , yt_b in self.test_loader: 
                    xt_b = xt_b.to(self.device)
                    yt_b = yt_b.to(self.device)
                    
                    outputs = self.model(xt_b)
                    predictions = torch.argmax(outputs,dim=1)
                    correct += (predictions == yt_b).sum().item()
                    total += yt_b.size(0)
            accuracy = correct / total
            self.test_accuracies.append(accuracy)
            
            
            avg_training_loss = torch.tensor(self.train_losses,dtype = torch.float32).mean()
            avg_testing_accuracy = torch.tensor(self.test_accuracies,dtype = torch.float32).mean()
            
        print(f'Average Training Loss : {avg_training_loss}')
        print(f'Average Testing Accuracy : {avg_testing_accuracy}')

    def save_model(self,path="/Users/sayar/Desktop/canvas/MNSIT CANVAS/model.pth"): 
        torch.save(self.model.state_dict(),path)

    def plot(self): 
        plt.figure()
        plt.plot(self.train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        
        plt.figure()
        plt.plot(self.test_accuracies)
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

model = MNSITPYTORCH(df) 

model.looping() 
model.save_model()
model.plot() 