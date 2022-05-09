import torch #python #keras #tensorflow #pytorch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils import shuffle
from torchsummary import summary

# class Net(touch.nn.model)
  # hidden layer, activation layes and output function, optimizer
class Net(torch.nn.Module):
    def __init__(self, num_inputs, size_hidden_1, size_hidden_2, n_output):
        super(Net, self).__init__()
    
        self.hidden_layer_1_a = torch.nn.Linear(num_inputs, size_hidden_1)    # hidden layer
        self.activation_1_a = torch.nn.Tanh() # activation layer
        self.hidden_layer_1_b = torch.nn.Linear(num_inputs, size_hidden_2)   # hidden layer
        self.activation_1_b = torch.nn.Tanh() # activation layer
        self.output_layer = torch.nn.Linear(size_hidden_1+size_hidden_2, n_output)   # output layer
        self.output_act = torch.nn.Sigmoid()

    def forward(self, x):
        a = self.activation_1_a(self.hidden_layer_1_a(x))      # activation function for hidden layer
        b = self.activation_1_b(self.hidden_layer_1_b(x))      # activation function for hidden layer
        x = torch.cat((a,b),-1)    # activation function for hidden layer
        x = self.output_act(self.output_layer(x))                    # output
        return x

# variables to be observed : batch_size, epoch, learning rate, size of hidder_layers, no of hidder layes, 
#                            batch no, cols/features, target/classes, the type of functions required

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()



#Define training hyperprameters.
batch_size = 1000
num_epochs = 50
learning_rate = 0.02
size_hidden_1 = 100
size_hidden_2 = 100
num_classes = 3

#Calculate some other hyperparameters based on data.  
batch_no = len(X_train) // batch_size  #batches
cols = X_train.shape[1] #Number of columns in input matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net(cols, size_hidden_1, size_hidden_2, num_classes)
summary(net, (1,cols))

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()  

for epoch in range(num_epochs):
    #Shuffle just mixes up the dataset between epocs
    X_train, y_train = shuffle(X_train, y_train)

    train_acc = 0.0
    running_loss = 0.0

    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(X_train[start:end]))
        labels = Variable(torch.LongTensor(y_train[start:end]))
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #print("outputs",outputs,outputs.shape,"labels",labels, labels.shape)
        #loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
        loss = loss_func(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        acc = get_accuracy(outputs, labels, batch_size)
        train_acc += acc
      
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch+1, running_loss / (i+1), train_acc/(i+1)))  
    running_loss = 0.0

y_pred = net(Variable(torch.FloatTensor(X_test)))
accuracy = get_accuracy(y_pred, Variable(torch.LongTensor(y_test)), len(y_test))
print(accuracy)