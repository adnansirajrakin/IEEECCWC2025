import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


class _quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output/ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply

class quan_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return  F.linear(input, self.weight*self.step_size, self.bias)
        else: 
            weight_quan = quantize(self.weight, self.step_size,
                               self.half_lvls)*self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max()/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True    


def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
  
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if  isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).char()
            
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return
        

        
# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = quan_Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = quan_Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
'''for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))'''


pretrained_dict = torch.load('model.pt')
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)
n=0
# update the step size before validation
for m in model.modules():
    if isinstance(m, quan_Linear):
        n=n+1
        #print(m.weight.size(),n)  
        m.__reset_stepsize__()
        m.__reset_weight__()


weight_conversion(model)
model.eval()
#torch.save(model.state_dict(), 'model.pt')

#model.load_state_dict(torch.load('model.pkl'))

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.pkl')
