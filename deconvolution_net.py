import torch
from skimage import color, data
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import numpy as np

dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

K_SIZE = 13 # PSF size
PADDING = 6 # this should be K_SIZE/2, rounded towards 0
IM_SIZE = 171 # image size

im_np = resize(color.rgb2gray(data.astronaut()), [IM_SIZE,IM_SIZE], mode='constant')
im_np = im_np/np.amax(im_np)
plt.figure(figsize=(6, 6))
plt.gray()
plt.imshow(im_np,vmin=0,vmax=1)
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


print(im_np.sum())

im = torch.from_numpy(im_np).float().to(device = device) 
psf = torch.from_numpy(psf_np).float().to(device = device) 

#print('im shape:', im.shape)

im = im.expand([1,1,IM_SIZE,IM_SIZE]) 
#don't know why, but I was not able to do the convolution without adding 2 channels with 1 single element 
psf = psf.expand([1,1,K_SIZE,K_SIZE])

#generate blurred images with torch.nn.functional.conv2d
im_blurred = torch.nn.functional.conv2d(im, psf, bias=None, stride=1, padding=PADDING, dilation=1, groups=1).to(device = device)  

#add noise
#im_blurred += 0.1* torch.randn(im_blurred.shape).to(device = device) 

#print('\n im_blurred + noise mean:')
#print(torch.mean(im_blurred))

plt.figure(figsize=(6, 6))
plt.imshow(im_blurred.squeeze().cpu(),vmin=0,vmax=1) #opposite of expand
plt.title('Blurred image');
plt.pause(0.05)

"""
                        Deconvolution starts here

"""

net = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(1,1,IM_SIZE,1,PADDING).to(device = device)
    # other layers could be added here, separated by comma
)

print(net)

params = list(net.parameters())
print('\nNumber of parameters:', len(params), '\nSize of the first parameter:', params[0].size(),'\n')

# Define a loss function
loss_fn = torch.nn.MSELoss(reduction='sum').to(device = device)
#loss_fn = torch.nn.SmoothL1Loss()

# Use the optim package to define an Optimizer that will update the weights of the model for us. 
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.6)

for t in range(50):
    '''
    Forward pass: compute predicted blurred image by passing the psf.
    The reconstructed image is stored inside the net and can be called in one 
    of the two following ways:
        net[0].weight
        list(net.parameters())[0].data
    The net has also a bias (of size 1):
        net[1].bias
        list(net.parameters())[1].data
    '''    
    im_pred = net(psf)
    
    # Compute and print loss.
    loss = loss_fn(im_pred, im_blurred)
        
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    
    if t % 5 == 4: #print loss and show image only sometimes
        
        print('step:',t,
              ',loss:', loss.item()
             )
        
        plt.figure(figsize=(6, 6))
        plt.title('Deconvolved image, step:' + str(t));  
        plt.imshow(list(net.parameters())[0].data.squeeze().cpu().numpy(),
                   vmin=0,
                   vmax=1)
        plt.pause(0.05)

plt.figure(figsize=(6, 6))
plt.imshow(net[0].weight.squeeze().detach().cpu().numpy(),vmin=0,vmax=1)
plt.title('Deconvolved image (final)');

del net,im,im_blurred,im_pred
if torch.cuda.is_available():
    torch.cuda.empty_cache()