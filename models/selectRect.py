import cv2
import os
import glob
import random
import numpy as np

k = 48
SCREENTONE_DIR = "Flow2/3.5/sc"
MARK_DIR = "Flow2/3.5/mark" + str(k)

class SelectAvailableRange:
    
    def __init__(self, k):
        self.k = k
        self.kh = k // 2
        self.kh2 = k - self.kh
        self.k_2 = k * k

    def getRandomXY(self, imgPath):
        result = self.compute(imgPath)
        if result:
            mark, minx, miny, maxx, maxy = result
            while True:
                x = random.randint(minx, maxx)
                y = random.randint(miny, maxy)
                if mark[y, x] == 255:
                    return x, y
    
    def compute(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            _, _, channel = img.shape
            if channel == 4:
                img = img[:, :, 3]
            else:
                img = img[:, :, 0]
        else:
            imgHeight, imgWidth = img.shape
            return (np.ones(img.shape, np.uint8) * 255, self.kh, self.kh, imgWidth-self.kh, imgHeight-self.kh)
        
        imgHeight, imgWidth = img.shape
        
        def testXY(x, y):
            nonlocal img
            xs = x - self.kh
            ys = y - self.kh
            xe = x + self.kh2
            ye = y + self.kh2
            if mark[y, x - 1] == 255:
                tmpImg = img[ys:ye, xe].reshape([self.k]) == 255
                return 0 if (self.k - np.count_nonzero(tmpImg)) == 0 else self.k_2
            elif mark[y - 1, x] == 255:
                tmpImg = img[ye, xs:xe].reshape([self.k]) == 255
                return 0 if (self.k - np.count_nonzero(tmpImg)) == 0 else 1
            else:
                tmpImg = img[ys:ye, xs:xe].reshape([self.k*self.k]) == 255
                return self.k_2 - np.count_nonzero(tmpImg)

        ix, iy, iw, ih = cv2.boundingRect(img)
        
        minx, miny, maxx, maxy = imgWidth, imgHeight, 0, 0
        def updateMinMax(x, y):
            nonlocal minx, miny, maxx, maxy
            minx = x if x < minx else minx
            miny = y if y < miny else miny
            maxx = x if x > maxx else maxx
            maxy = y if y > maxy else maxy
        # print(abs(rect[0] - rect[1]), abs(rect[2] - rect[3]), rect)
        if iw >= self.k and ih >= self.k:
            count = 0
            mark = np.zeros((imgHeight, imgWidth, 1), np.uint8)
            for y in range(iy + self.kh, iy + ih - self.kh2):
                skip = 0
                for x in range(ix + self.kh, ix + iw - self.kh2):
                    if skip > 0:
                        skip -= 1
                        continue
                    diff = testXY(x, y)
                    if diff == 0:
                        mark[y, x] = 255
                        count += 1
                        updateMinMax(x, y)
                    else:
                        skip = diff // self.k - 1
            if count > 0:
                print(imgPath, "found: ", count)
                return (mark, minx, miny, maxx, maxy)
        return None
    
    def check(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            _, _, channel = img.shape
            if channel == 4:
                img = img[:, :, 3]
            else:
                img = img[:, :, 0]
        else:
            return True
        imgHeight, imgWidth = img.shape
        
        def testXY(x, y):
            nonlocal img
            xs = x - self.kh
            ys = y - self.kh
            xe = x + self.kh2
            ye = y + self.kh2
            if mark[y, x - 1] == 255:
                tmpImg = img[ys:ye, xe].reshape([self.k]) == 255
                return 0 if (self.k - np.count_nonzero(tmpImg)) == 0 else self.k_2
            elif mark[y - 1, x] == 255:
                tmpImg = img[ye, xs:xe].reshape([self.k]) == 255
                return 0 if (self.k - np.count_nonzero(tmpImg)) == 0 else 1
            else:
                tmpImg = img[ys:ye, xs:xe].reshape([self.k*self.k]) == 255
                return self.k_2 - np.count_nonzero(tmpImg)
        
        ix, iy, iw, ih = cv2.boundingRect(img)
        if iw >= self.k and ih >= self.k:
            mark = np.zeros((imgHeight, imgWidth, 1), np.uint8)
            for y in range(iy + self.kh, iy + ih - self.kh2):
                skip = 0
                for x in range(ix + self.kh, ix + iw - self.kh2):
                    if skip > 0:
                        skip -= 1
                        continue
                    diff = testXY(x, y)
                    if diff == 0:
                        return True
                    else:
                        skip = diff // self.k - 1
        return False

def checkAndMakeDir(dir):
    if not os.path.isdir(dir):
        print("create folder:", dir)
        os.mkdir(dir)

if __name__ == "__main__":

    checkAndMakeDir(MARK_DIR)

    selecter = SelectAvailableRange(k)
    scs = glob.glob(os.path.join(SCREENTONE_DIR, "**/*.png"))
    for sc in scs:
        result = selecter.compute(sc)
        if result:
            mark = result[0]
            scType = int(os.path.basename(os.path.split(sc)[0]))
            scDir = os.path.join(MARK_DIR, str(scType))
            checkAndMakeDir(scDir)
            output = os.path.join(scDir, os.path.split(sc)[1])
            print("Write file:", output)
            cv2.imwrite(output, mark)
