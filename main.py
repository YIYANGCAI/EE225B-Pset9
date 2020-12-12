from scipy.io import loadmat
import numpy as np
import cv2 as cv

class fftReconstruction(object):
    def __init__(self):
        super(fftReconstruction).__init__()
    
    def fft_original(self, img):
        #import pdb;pdb.set_trace()
        if len(img.shape) > 2: img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        rows, cols = img.shape
        img = img / 255.0
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        phase = np.angle(fshift)
        magnitude = np.abs(fshift)

        f_ishift = np.fft.ifftshift(magnitude*np.e**(1j*phase))
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = img_back * 255.0
        img_back = img_back.astype('uint8')
        return img_back
    
    def reconstructFromPhase(self, phase_data, iteration, init_data):
        '''
        phase_data: the phase data of original image
        iteration: time of iteration
        init_data: initialized space domain image
        '''
        h, w = init_data.shape[0], init_data.shape[1]
        img_previous = None
        for i in range(iteration):
            #print("Iteration {}".format(i))
            if i == 0:
                input_data = init_data
            else:
                input_data = img_previous
            # dft process
            input_data = input_data / 255.0
            f = np.fft.fft2(input_data)
            fshift = np.fft.fftshift(f)
            # obtain the phase and magnitude
            phase = np.angle(fshift)
            #residual = np.sum(phase - phase_data)
            #print(residual)
            magnitude = np.abs(fshift)
            # replace the original phase with target phase
            img_mandp = (magnitude+2) * np.e**(1j*phase_data)
            # idft process
            f_ishift = np.fft.ifftshift(img_mandp)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = img_back * 255.0
            img_back = img_back.astype('uint8')
            # set the external region to zero
            img_back[int(h/2):,:] = 0
            img_back[:,int(w/2):] = 0
            img_previous = img_back
        return img_back

    def reconstructFromMagnitude(self, magnitude_data, iteration, init_data):
        '''
        magnitude_data: the magnitude data of original image
        iteration: time of iteration
        init_data: initialized space domain image
        '''
        h, w = init_data.shape[0], init_data.shape[1]
        img_previous = None
        for i in range(iteration):
            #print("Iteration {}".format(i))
            if i == 0:
                input_data = init_data
            else:
                input_data = img_previous
            # dft process
            input_data = input_data / 255.0
            f = np.fft.fft2(input_data)
            fshift = np.fft.fftshift(f)
            # obtain the phase and magnitude
            phase = np.angle(fshift)
            #residual = np.sum(phase - phase_data)
            #print(residual)
            magnitude = np.abs(fshift)
            # replace the original phase with target phase
            img_mandp = magnitude_data * np.e**(1j*phase)
            # idft process
            f_ishift = np.fft.ifftshift(img_mandp)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = img_back * 255.0
            img_back = img_back.astype('uint8')
            # set the external region to zero
            img_back[int(h/2):,:] = 0
            img_back[:,int(w/2):] = 0
            img_previous = img_back
        return img_back

def main():
    s = fftReconstruction()

    # problem1 - phase only
    ## obtain the target phase
    data = loadmat('Phase.dat')
    phase = np.angle(data['ImagePhase'])
    ## set the initial data
    h, w = phase.shape[0], phase.shape[1]
    init_img = np.random.rand(h,w)*255
    init_img[int(h/2):,:] = 0
    init_img[:,int(w/2):] = 0

    out = s.reconstructFromPhase(phase, 50, init_img)[:int(h/2),:int(w/2)]
    cv.imwrite('result_1.jpg', out)

    # problem2 - magnitude only
    ## obtain the target phase
    data = loadmat('Magnitude.dat')
    magnitude = np.abs(data['ImageMagnitude'])
    ## set the initial data
    h, w = magnitude.shape[0], magnitude.shape[1]
    init_img = np.random.rand(h,w)*255
    init_img[int(h/2):,:] = 0
    init_img[:,int(w/2):] = 0

    out = s.reconstructFromMagnitude(magnitude, 50, init_img)[:int(h/2),:int(w/2)]
    cv.imwrite('result_2.jpg', out)


if __name__ == "__main__":
    main()