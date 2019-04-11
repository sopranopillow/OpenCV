import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint
'''
# manipulating matrices-creating, filling, accessing elements, and ROIs
image = np.full((480, 640, 3), 255, np.uint8)
cv2.imshow('white', image)
cv2.waitKey()
cv2.destroyAllWindows()

image = np.full((480, 640, 3), (0, 0, 255), np.uint8)
cv2.imshow('red', image)
cv2.waitKey()
cv2.destroyAllWindows()

image.fill(0)
cv2.imshow('black', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[240, 160] = image[240, 320] = image[240, 480] = (255, 255, 255)
cv2.imshow('black with white pixels', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:, :, 0] = 255
cv2.imshow('blue with white pixels', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:, 320, :] = 255
cv2.imshow('blue with white line', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[100:600, 100:200, 1] = 255
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#Converting between data types and scaling values
image = cv2.imread('../../Pictures/dog.jpg')
print('Shape:', image.shape)
print('Data type:', image.dtype)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

image = image.astype(np.float32) / 50
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', np.clip(image*2, 0, 1))
cv2.waitKey()
cv2.destroyAllWindows()

image = (image * 255).astype(np.uint8)
print('Shape:', image.shape)
print('Data type:', image.dtype)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#non-image data persistence
mat = np.random.rand(100, 100).astype(np.float32)
print('Shape:', mat.shape)
print('Data type:', mat.dtype)

np.savetxt('mat.csv', mat)

mat = np.loadtxt('mat.csv').astype(np.float32)
print('Shape:', mat.shape)
print('Data type:', mat.dtype)
'''
'''
#Manipulating image channels
image = cv2.imread('../../Pictures/dog.jpg').astype(np.float32) / 255
print('Shape:', image.shape)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:, :, [0, 2]] = image[:, :, [2, 0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:, :, [0, 2]] = image[:, :, [2, 0]]
image[:, :, 0] = (image[:, :, 0] * 0.9).clip(0, 1)
image[:, :, 1] = (image[:, :, 1] * 1.1).clip(0, 1)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#converting images from one color space to another
image = cv2.imread('../../Pictures/dog.jpg').astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.dtype)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape:', gray.shape)
print('Data type:', gray.dtype)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape:', hsv.shape)
print('Data type:', hsv.dtype)
cv2.imshow('hsv', hsv)
cv2.waitKey()
cv2.destroyAllWindows()

hsv[:, :, 2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape:', from_hsv.shape)
print('Data type:', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#removing noise using Gaussian, median, and bilateral filters
image = cv2.imread('../../Pictures/dog.jpg').astype(np.float32) / 255
noised = (image + 0.2 *
np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
plt.imshow(noised[:,:,[2,1,0]])
plt.show()

gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
plt.imshow(gauss_blur[:, :, [2, 1, 0]])
plt.show()

median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
plt.imshow(median_blur[:, :, [2, 1, 0]])
plt.show()

bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
plt.imshow(bilat[:, :, [2, 1, 0]])
plt.show()
'''
'''
# computing gradients using sobel operator
image = cv2.imread('../../Pictures/dog.jpg', 0)
dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')
plt.subplot(133)
plt.axis('off')
plt.title(r'$\frac{dI}{dy}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()
'''
'''
#creating and applying a custom filter
image = cv2.imread('../../Pictures/dog.jpg')

KSIZE = 200
ALPHA = 3
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

filtered = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout(True)
plt.show()
'''
'''
#processing images with real-valued gabor filters
image = cv2.imread('../../Pictures/dog.jpg', 0).astype(np.float32) / 255
kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())
filtered = cv2.filter2D(image, -1, kernel)
plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()
'''
'''
#going from the spatial domain to the frequency domain(and back) using the discrete fourier transform
image = cv2.imread('../../Pictures/dog.jpg', 0).astype(np.float32) / 255
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
shifted = np.fft.fftshift(fft, axes=[0, 1])
magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
magnitude = np.log(magnitude)
plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout()
plt.show()
restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
'''
'''
#Processing images with different thresholds
image = cv2.imread('../../Pictures/dog.jpg', 0)
thr, mask = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY)
print('Threshold used:', thr)
adapt_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('binary threshold')
plt.imshow(mask, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('adaptive threshold')
plt.imshow(adapt_mask, cmap='gray')
plt.tight_layout()
plt.show()
'''
'''
#Morphological operators
image = cv2.imread('../../Pictures/me.jpg', 0)
_, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3, 3),
iterations=10)
dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3, 3),
iterations=10)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
iterations=5)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
iterations=5)
grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
plt.figure(figsize=(10,10))
plt.subplot(231)
plt.axis('off')
plt.title('binary')
plt.imshow(binary, cmap='gray')
plt.subplot(232)
plt.axis('off')
plt.title('erode 10 times')
plt.imshow(eroded, cmap='gray')
plt.subplot(233)
plt.axis('off')
plt.title('dilate 10 times')
plt.imshow(dilated, cmap='gray')
plt.subplot(234)
plt.axis('off')
plt.title('open 5 times')
plt.imshow(opened, cmap='gray')
plt.subplot(235)
plt.axis('off')
plt.title('close 5 times')
plt.imshow(closed, cmap='gray')
plt.subplot(236)
plt.axis('off')
plt.title('gradient')
plt.imshow(grad, cmap='gray')
plt.tight_layout()
plt.show()
'''
'''
#extracting connected components from a binary image
img = cv2.imread('../../Pictures/me.jpg', cv2.IMREAD_GRAYSCALE)
connectivity = 8
num_labels, labelmap = cv2.connectedComponents(img, connectivity,
cv2.CV_32S)
img = np.hstack((img, labelmap.astype(np.float32)/(num_labels -
1)))
cv2.imshow('Connected components', img)
cv2.waitKey()
cv2.destroyAllWindows()
img = cv2.imread('../../Pictures/me.jpg', cv2.IMREAD_GRAYSCALE)
otsu_thr, otsu_mask = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
output = cv2.connectedComponentsWithStats(otsu_mask, connectivity,
cv2.CV_32S)
num_labels, labelmap, stats, centers = output
colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)
for l in range(1, num_labels):
    if stats[l][4] > 200:
        colored[labelmap == l] = (0, 255*l/num_labels,255*num_labels/l)
        cv2.circle(colored,(int(centers[l][0]), int(centers[l][1])), 5,(255, 0, 0), cv2.FILLED)
img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)
cv2.imshow('Connected components', np.hstack((img, colored)))
cv2.waitKey()
cv2.destroyAllWindows()
'''
#image segmentation using segment seeds - the watershed algorithm
img = cv2.imread('../../Pictures/me.jpg')
show_img = np.copy(img)
seeds = np.full(img.shape[0:2], 0, np.int32)
segmentation = np.full(img.shape, 0, np.uint8)

n_seeds = 9
colors = []
for m in range(n_seeds):
    colors.append((255 * m / n_seeds, randint(0, 255), randint(0,255)))
mouse_pressed = False
current_seed = 1
seeds_updated = False

def mouse_callback(event, x, y, flags, param):
    global mouse_pressed, seeds_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(seeds, (x, y), 5, (current_seed), cv2.FILLED)
        cv2.circle(show_img, (x, y), 5, colors[current_seed - 1],cv2.FILLED)
        seeds_updated = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(seeds, (x, y), 5, (current_seed),cv2.FILLED)
            cv2.circle(show_img, (x, y), 5, colors[current_seed - 1], cv2.FILLED)
            seeds_updated = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
while True:
    cv2.imshow('segmentation', segmentation)
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('c'):
        show_img = np.copy(img)
        seeds = np.full(img.shape[0:2], 0, np.int32)
        segmentation = np.full(img.shape, 0, np.uint8)
    elif k > 0 and chr(k).isdigit():
        n = int(chr(k))
        if 1 <= n <= n_seeds and not mouse_pressed:
            current_seed = n
    if seeds_updated and not mouse_pressed:
        seeds_copy = np.copy(seeds)
        cv2.watershed(img, seeds_copy)
        segmentation = np.full(img.shape, 0, np.uint8)
        for m in range(n_seeds):
            segmentation[seeds_copy == (m + 1)] = colors[m]
        seeds_updated = False
cv2.destroyAllWindows()