from cv2 import threshold
from NoiseGenerator import generate
import random
import numpy as np
import cv2 

def getThresholdRange(img):
    row, col = img.shape
    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    black, white = 0, 0
    
    sum = 0
    for i in range(256):
        sum += hist[i]
        black = i
        if sum / (row * col) > 0.2:
            break
    sum = 0
    for i in range(255, -1, -1):
        sum += hist[i]
        white = i
        if sum / (row * col) > 0.2:
            break
    return black, white

def randMinMore(min, max):
    return abs(np.random.normal(0, max - min)) + min

def genArg(row=32, col=32, gen_scale=1.5):
    arg = {
        "row": row * gen_scale,
        "col": col * gen_scale,
        "seed": random.randint(-2147483648, 2147483647),
        "frequency": np.random.normal(0.18, 0.04) / gen_scale,
        "noise_type": random.randrange(0, 6),
        "fractal_type": random.randrange(0, 6),
        "octaves": randMinMore(1, 8),
        "lacunarity": randMinMore(0, 15),
        "gain": randMinMore(0, 4),
        "weighted_strength": randMinMore(0, 5),
        "ping_pong_strength": randMinMore(0, 8),
        "cellular_distance_func": random.randrange(0, 4),
        "cellular_return_type": random.randrange(0, 7),
        "cellular_jitter_mod": np.random.normal(0.16, 0.04) / gen_scale,
        
        "threshold": random.random(),
        "invert": random.randint(0, 1) == 0,
    }
    return arg
    
def genImg(arg, row=32, col=32, gen_scale=1.5):
    data = generate(arg)
    data = np.array(data)
    data = ((data + 1) / 2 * 256)
    data = np.reshape(data, (int(row * gen_scale), int(col * gen_scale)))

    # threshold
    min_t, max_t = getThresholdRange(data)
    thres = arg["threshold"] * (max_t - min_t) + min_t
    _, data = cv2.threshold(data, thres, 255, cv2.THRESH_BINARY)

    # inverse
    if arg["invert"]:
        data = (255 - data)
    data = cv2.resize(data, (col, row))
    return data
    
if __name__ == "__main__":
    arg = genArg()
    img = genImg(arg)
    cv2.imwrite("output.png", img)