import torch
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10


#t = np.random.randn(D_in)
t = np.linspace(-1,1,D_in)
f = np.linspace(0,5,N)

x_np = np.zeros([N, D_in])

l=enumerate(f)



for indx,val in enumerate(f):
    freq=f[indx]
    #print(freq)
    x_np[indx,:] = np.sin(2*np.pi*freq*t)

# Create random input and output data
#x_np = np.random.randn(N, D_in)
_y = np.abs((np.fft.fft((x_np))))

y_np = _y[:,0:D_out]
#y_np = y_np /np.amax(y_np) 


#y = np.random.randn(N, D_out)
#y= y/np.amax(y) 
#print(y.shape)

f_index = 30

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

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.title('reconstructed y at frequency: '+ str(f[f_index]))
plt.plot(y_pred[f_index])
plt.show()
