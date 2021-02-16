'''I used google colab for the code'''
!gdown https://drive.google.com/uc?id=101sgazgxxT1bUDyTLVYlr3i1H2VCAd2m # Control
!gdown https://drive.google.com/uc?id=1D1XbF5SdQ8lJNb_IFyYbHMI5w7ukY8xs #W/ magnets
# - - - - - - - - - (dividing cell from cell)
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh #doh (Determinant of Hessian) does light over dark, whereas log and dog detect dark on light
filname = '/content/20201030_171546_002.mp4'
vid = imageio.get_reader(filname, "mp4")
frames_stack = []
blobs1 = []
for frame in vid.iter_data():
  frame = frame[500:750,400:800,0]
  frames_stack.append(frame)
  blob1 = np.array(blob_doh(frame, min_sigma= 10, max_sigma = 50, threshold = 0.01)).squeeze()
  blobs1.append(blob1) 
blobs1 = np.array(blobs1)
frames_stack = np.array(frames_stack)
print (blobs1.shape)
print (frames_stack.shape) 
plt.imshow(frames_stack[100,:,:])
# - - - - - - - - - - - - - - - - - - - 
from scipy.optimize import curve_fit
x = []
y = []
frames = []
i = 0
for blob in blobs:
  frames.append(i/240)
  i += 1
  x.append(blob[1])
  y.append(-blob[0] + 810)
x = np.array(x)
y = np.array(y)
plt.scatter(frames, x, label = 'data') #Plotting x and y separate from function fit
plt.xlabel("Time (s)")
plt.ylabel("X-position of the Pendulum (Pixels)")
plt.title("X-position of Pendulum over time)")
plt.show()
plt.scatter(frames, y, label = 'data')
plt.xlabel("Time (s)")
plt.ylabel("Y-position of Pendulum (Pixels)")
plt.title("Y-position of Pendulum over time")
plt.show()
def func(z, a, b, c, d, e):
  return a * np.cos(b*(z + c)) + d*z + e  #Function template you use heavily depends on trend you observe for X over time graph
popt, pcov = curve_fit(func, frames, x, p0 = [-150, 6, 0.2, 10, 500]) #Initial guess p0 also heavily depends on trend you observe for X over time graph
print (popt)
plt.scatter(frames, x, label = 'data')
plt.plot(frames, [func(frame, a = popt[0], b = popt[1], c = popt[2], d = popt[3], e = popt[4]) for frame in frames], 'r', label = "Fitted sinusoidal function")
plt.xlabel("Time (s)")
plt.ylabel("X-position of the Pendulum (Pixels)")
plt.title("X-position of Pendulum over time)")
plt.legend(loc = 'upper right')
plt.show()
plt.scatter(frames, y, label = 'data')
plt.xlabel("Time (s)")
plt.ylabel("Y-position of Pendulum (Pixels)")
plt.title("Y-position of Pendulum over time")
def func1(z, a, b, c, d, e, f): #Function template you use heavily dependent on trend you observe for X over time graph
  return a * (np.sin(2*b*z + c))**2 + d * (np.sin(b*z)) * (np.cos(b*z + c)) + e * z + f
popt1, pcov1 = curve_fit(func1, frames, y) #No p0 needed here, but you might need it for your own experiment
print (popt1)
plt.plot(frames, [func1(frame, a = popt1[0], b = popt1[1], c = popt1[2], d = popt1[3], e = popt1[4], f = popt1[5]) for frame in frames], 'r', label = 'fitted function')
plt.legend(loc = 'upper right')
plt.show()
