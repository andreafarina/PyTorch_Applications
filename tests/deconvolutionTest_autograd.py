import torch

from skimage import color, data
from skimage.transform import rescale
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import numpy as np

dtype = torch.float
device = torch.device("cpu")

im = rescale(color.rgb2gray(data.astronaut()), 1.0/3.0, mode='constant')

plt.figure(figsize=(6, 6))
plt.gray()
plt.imshow(im)
plt.title('Astronaut image')



def gaussian(kernlen, nsig_x, nsig_y):
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    kern = fi.gaussian_filter(inp, (nsig_x, nsig_y))
    scaled_kern = kern / np.sum(kern)
    return scaled_kern


psf = gaussian(kernlen=13, nsig_x=1, nsig_y=3)

plt.figure(figsize=(6, 6))
plt.imshow(psf)
plt.title('Gaussian point spread function');

im = torch.from_numpy(im).float().to(device = device) 
psf = torch.from_numpy(psf).float().to(device = device) 

print(im.shape)

im = im.expand([1,1,171,171])
psf = psf.expand([1,1,13,13])

im_blurred = torch.nn.functional.conv2d(im, psf, bias=None, stride=1, padding=6, dilation=1, groups=1) 

plt.figure(figsize=(6, 6))
plt.imshow(im_blurred.squeeze())
plt.title('Gaussian point spread function');

print(im_blurred.shape)

learning_rate = 1e-6

print('psf_shape:')
print(psf.shape)

pt_input = torch.rand(1, 1, 171, 171, requires_grad=True)
#pt_input = torch.rand(1, 1, 171, 171)


print('pt_input_shape:')
print(pt_input.shape)


for t in range(5):
    # Forward pass: compute predicted y by passing x to the model.

    #y_pred = model(pt_input,psf)
    #would be better to use 
    
    im_pred = torch.nn.functional.conv2d(pt_input, psf, bias=None, stride=1, padding=6, dilation=1, groups=1) 

    #print('y_pred_shape:')
    #print(y_pred.shape)
    
    loss = (im_pred - im_blurred).pow(2).sum()
    # Compute and print loss.
    # loss = loss_fn(im_pred, im_blurred)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    #optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    with torch.no_grad():
        pt_input -= learning_rate * pt_input.grad
        

        # Manually zero the gradients after updating weights
        pt_input.grad.zero_()


#detach().numpy()

print(im_pred.shape)


#plt.figure(figsize=(6, 6))
#plt.imshow(y_pred.squeeze())

plt.figure(figsize=(6, 6))
plt.imshow(pt_input.squeeze().detach().numpy())
plt.title('Gaussian point spread function');