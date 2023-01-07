# Scale-Invariant Feature Transform
import cv2
import numpy as np
import math

class myDifferenceOfGaussianPyramid():
    r"""
    ClassName       :   高斯差分金字塔
    ClassDescribe   :   
    Specification   :   
    """

    class myOctave():
        r"""
        ClassName       :   
        ClassDescribe   :   
        Specification   :   
        """
        def __init__(self, octaveID, sliceNumOfOctave, octaveOriginImg, octaveFirstSigma, sigmaGainPerSlice, gaussianKernelSize):
            self.octaveID = octaveID  # octave的编号，小的在下，尺寸跟大
            self.sliceNumOfOctave = sliceNumOfOctave  # octave所含slice数量
            self.octaveOriginImg = octaveOriginImg  # 高斯卷积所用的原始图像
            self.octaveImgs = []  # 高斯卷积后的各个图像
            self.slicesShape = octaveOriginImg.shape  # slice尺寸
            self.octaveFirstSigma = octaveFirstSigma  # 初始sigma
            self.octaveSigmas = []  # 高斯卷积的各个sigma
            self.sigmaGainPerSlice = sigmaGainPerSlice  # slice间的sigma放大比例
            self.gaussianKernelSize = gaussianKernelSize  # 高斯卷积核大小
        
        def initSlices(self):
            tempSigma = self.octaveSigmas[0]
            tempKernelSize = self.get2DGaussianKernelSizeBySigma(tempSigma)
            for sliceID in range(self.sliceNumOfOctave):
                tempImg = cv2.GaussianBlur(self.octaveOriginImg, )
                (self.octaveImgs).append(tempSigma)
                (self.octaveSigmas).append(tempSigma)
                
                tempSigma = tempSigma*self.sigmaGainPerSlice  # 为下一次sigma做准备
            pass
        
        def get2DGaussianKernelSizeBySigma(sigma):  # 使用sigma反推kernelSize
            # 利用OpenCV函数： getGaussianKernel()中的ksize推sigma进行反推
            # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
            # 也有别的反推策略(×3+1、×6+1)
            kernelSize = int(((sigma-0.8)/0.3+1)*2+1)
            if kernelSize%2.0==0:  # 保证为奇数
                kernelSize = kernelSize+1
            return (kernelSize,kernelSize)



    def __init__(self, imgOrigin, octaveNumOfPyramid = None, wantedUsefulDOGs = None):
        r"""
        FunctionDescribe:   Init the DOGs Pyramid. Octaves模拟近大远小，高斯核卷积模拟清晰和模糊
        InputParameter  :   ①
                            ②octaveNumOfPyramid(>0) : 如果是None，则使用论文推荐值
                            ③wantedUsefulDOGs(>0)    : 一层octave差分后得到的DOGs中可用于三维特征提取的层数，其值+3即为一层octave所含的slices数量
                                                        +3的解释：首先DOGs的上下两层不能使用(无法求导找极值点)，所以+2;其次是差分所得，所以要再+1
        OutputParameter :   ①②③④⑤⑥⑦⑧⑨⑩
        Specification   :   Related Blog : https://www.bilibili.com/video/BV1Qb411W7cK/
        """
        if len(imgOrigin.shape)!=2:  # 图片维数必须为2
            print("[myDifferenceOfGaussianPyramid::__init__] : The img's dimension is not 2.")
            exit -1
        self.imgOrigin = imgOrigin

        if octaveNumOfPyramid is None :
            self.octaveNumOfPyramid = int(math.log2(min(imgOrigin.shape[0], imgOrigin.shape[1])))-3  # 金字塔大层数（论文推荐数量）
        elif octaveNumOfPyramid < 0:
            print("[myDifferenceOfGaussianPyramid::__init__] : octaveNumOfPyramid < 0.")
            exit -1

        if wantedUsefulDOGs is None:  # 没传入值，默认sliceNumOfOctave=10
            self.sliceNumOfOctave = 15  # 12 + 3
        elif wantedUsefulDOGs < 0:  # 给不合理的值
            print("[myDifferenceOfGaussianPyramid::__init__] : wantedUsefulDOGs < 0.")
            exit -1
        else:  # 正常wantedUsefulDOGs值 + 3得到sliceNumOfOctave
            self.sliceNumOfOctave = wantedUsefulDOGs + 3
        
        self.sigmaGainPerSlice = 2**(1/wantedUsefulDOGs)  # 同一Octave中，相邻slice的sigma增益
        self.allOctaves = []  # 创建Octaves数组
        self.pyramidFirstSigma = math.sqrt(1.6**2-0.5**2)  # 论文经验值
        pass

    def initOctaves(self):
        tempImg = self.imgOrigin
        for octaveID in range(self.octaveNumOfPyramid):  # idx小的在下
            tempOctave = myOctave()
            self.allOctaves

            tempImg = cv2.pyrDown(tempImg)  # 为上一层做准备，下采样，缩小为1/2
            pass

        pass


    pass




def MySIFT(img):
    r"""
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①②③④⑤⑥⑦⑧⑨⑩
    Specification   :   Related Blog : https://www.bilibili.com/video/BV1Qb411W7cK/
    """

    pass




if __name__ == "__main__":
    img = cv2.imread(r"./PicsForCode/StraightLines/StraightLines01.jpg")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img.shape)
    print(max(img.shape))
    print(max(img.shape[0], img.shape[1]))
    print(len(img.shape))


    gaussi

    pass
