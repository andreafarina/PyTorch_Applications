'''
Estimates the PSF given one blurred image and the corresponging ground truth image

'''

import torch
from skimage import color, data
from skimage.transform import rescale,resize
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import numpy as np

dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


K_SIZE = 15 # PSF size
PADDING = 0
IM_SIZE = 127 # image size

im_np = resize(color.rgb2gray(data.astronaut()), [IM_SIZE,IM_SIZE], mode='constant')

plt.figure(figsize=(6, 6))
plt.gray()
plt.imshow(im_np)
plt.title('Original image')

def gaussian(kernlen, nsig_x, nsig_y):
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    kern = fi.gaussian_filter(inp, (nsig_x, nsig_y))
    scaled_kern = kern / np.sum(kern)
    return scaled_kern

psf_np = gaussian(kernlen=K_SIZE, nsig_x=1, nsig_y=3)

plt.figure(figsize=(6, 6))
plt.imshow(psf_np)
plt.title('Gaussian point spread function');
plt.pause(0.05)

im = torch.from_numpy(im_np).float().to(device = device) 
psf = torch.from_numpy(psf_np).float().to(device = device) 

print(im.shape)

im = im.expand([1,1,IM_SIZE,IM_SIZE]) 
#don't know why, but I was not able to do the convolution without adding 2 channels with 1 single element 
psf = psf.expand([1,1,K_SIZE,K_SIZE])


#generate blurred images with torch.nn.functional.conv2d
im_blurred = torch.nn.functional.conv2d(im, psf, bias=None, stride=1, padding=PADDING, dilation=1, groups=1).to(device = device)  

plt.figure(figsize=(6, 6))
plt.imshow(im_blurred.squeeze().cpu()) #contrario di expand
plt.title('Blurred image');

print('\n im_blurred_mean:')
print(torch.mean(im_blurred))

#add noise
#im_blurred += torch.randn(im_blurred.shape).to(device = device) 

#print('\n im_blurred + noise mean:')
#print(torch.mean(im_blurred))

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Conv2d(1,1,K_SIZE,1,PADDING), 
    ).to(device = device)  

print(model)

loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.SmoothL1Loss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#optimizer = torch.optim.ASGD(model.parameters(), lr=1e-6)


#here the weights that are optimized are the coeffient of the kernel, 
#these are not even shown in the code
#they can be extracted as: model[0].weight.data
for t in range(10000):
    # Forward pass: compute predicted y by passing x to the model.
   
    im_pred = model(im)
    # Compute and print loss.
    loss = loss_fn(im_pred, im_blurred)
    if t % 100 == 99:
        print(t, loss.item())
        plt.figure(figsize=(6, 6))
        
        weight = model[0].weight.data.cpu().numpy()
        plt.imshow(weight[0, ...].squeeze())
        #plt.imshow(im_pred.squeeze().detach().cpu().numpy())
        plt.pause(0.05) 
        params = list(model.parameters())
        print(len(params))
        print(params[0].size())

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
 
#Extract weights that have been optimized
weight = model[0].weight.data.cpu().numpy()
plt.imshow(weight[0, ...].squeeze())
plt.title('Predicted PSF');

plt.figure(figsize=(6, 6))
plt.imshow(im_pred.squeeze().detach().cpu().numpy())
plt.title('Predicted image');