import torch
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 2560, 1000, 100, 50


#t = np.random.randn(D_in)
time = np.linspace(-1,1,D_in)
f = np.linspace(0,5,N)

x_np = np.zeros([N, D_in])


for indx,val in enumerate(f):
    freq=f[indx]
    #print(freq)
    x_np[indx,:] = np.sin(2*np.pi*freq*time)

# Create random input and output data
#x_np = np.random.randn(N, D_in)
_y = np.abs((np.fft.fft((x_np))))

y_np = _y[:,0:D_out]
#y_np= y_np/np.amax(y_np) 


#y = np.random.randn(N, D_out)
#y= y/np.amax(y) 
#print(y.shape)

f_index = 60

plt.plot(x_np[:][f_index])
plt.title('x at frequency: '+ str(f[f_index]))
plt.show()


plt.plot(y_np[:][f_index])
plt.title('y (fft of x) at frequency: '+ str(f[f_index]))
plt.show()

x = torch.from_numpy(x_np).float().to(device = device)

y = torch.from_numpy(y_np).float().to(device = device) 


# Create random input and output data
#x = torch.randn(N, D_in, device=device, dtype=dtype)
#y = torch.randn(N, D_out, device=device, dtype=dtype)


print(x.shape)

#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)



# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-2
#Note: here I am using Adam optimizer, and learning_rate is reduced from 1e-6 to 1e-3
#NOte that nomralizing y to 1 does not affect convergence 


# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

plt.title('reconstructed y at frequency: '+ str(f[f_index]))
plt.plot(y_pred.detach().numpy()[f_index])
plt.show()



x_test = np.sin(2*np.pi*2*time)

x_test = time*0

x_test[0:10] = 1



plt.title('x at frequency: 2.')
plt.plot(x_test)
plt.show()

x_T_test = torch.from_numpy(x_test).float().to(device = device)

y_model = model(x_T_test)
print(y_model.shape)

plt.title('reconstructed y at frequency: 2.')
plt.plot(y_model.detach().numpy())
plt.show()

