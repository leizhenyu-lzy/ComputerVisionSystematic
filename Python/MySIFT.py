# Scale-Invariant Feature Transform
import cv2
import numpy as np
import math

class MyDifferenceOfGaussianPyramid():
    """
    ClassName       :   高斯差分金字塔 \n
    ClassDescribe   :   \n
    Specification   :   \n
    """

    def __init__(self, imgOrigin, octaveNumOfPyramid = None, wantedUsefulDOGs = None, T=0.04, visualGenerate = False):
        """
        FunctionDescribe:   
                            Init the DOGs Pyramid. Octaves模拟近大远小，高斯核卷积模拟清晰和模糊 \n
        InputParameter  :   
                            ① \n
                            ②octaveNumOfPyramid(>0) : 如果是None，则使用论文推荐值 \n
                            ③wantedUsefulDOGs(>0)   : 一层octave差分得到的DOGs中可用于三维特征提取的层数，其值+3即为一层octave所含的slices数量 \n 
                                                        +3的解释：首先DOGs的上下两层不能使用(无法求导找极值点)，所以+2;其次是差分所得，所以要再+1 \n
        OutputParameter :   
                            ①②③④⑤⑥⑦⑧⑨⑩ \n
        Specification   :   
                            Related Blog : https://www.bilibili.com/video/BV1Qb411W7cK/ \n
        """
        if len(imgOrigin.shape)!=2:  # 图片维数必须为2
            print("[myDifferenceOfGaussianPyramid::__init__] : The img's dimension is not 2.")
            exit(-1)
        self.imgOrigin = imgOrigin

        if octaveNumOfPyramid is None :
            self.octaveNumOfPyramid = int(math.log2(min(imgOrigin.shape[0], imgOrigin.shape[1])))-3  # 金字塔大层数（论文推荐数量）
        elif octaveNumOfPyramid < 0:
            print("[myDifferenceOfGaussianPyramid::__init__] : octaveNumOfPyramid < 0.")
            exit(-1)

        if wantedUsefulDOGs is None:  # 没传入值，默认 sliceNumOfOctave = 10
            self.wantedUsefulDOGs = 12
            self.sliceNumOfOctave = 15  # 12 + 3
        elif wantedUsefulDOGs < 0:  # 给不合理的值
            print("[myDifferenceOfGaussianPyramid::__init__] : wantedUsefulDOGs < 0.")
            exit(-1)
        else:  # 正常wantedUsefulDOGs值 + 3得到sliceNumOfOctave
            self.wantedUsefulDOGs = wantedUsefulDOGs
            self.sliceNumOfOctave = self.wantedUsefulDOGs + 3
        
        self.sigmaGainPerSlice = 2**(1/wantedUsefulDOGs)  # 同一Octave中，相邻slice的sigma增益（论文推荐值）
        self.sigmaGainPerOctave = 2  # 相邻octave的sigma增益（论文推荐值，self.sigmaGainPerSlice的wantedUsefulDOGs次方，即为2）
        self.allOctaves = []  # 创建Octaves数组
        self.pyramidFirstSigma = math.sqrt(1.6**2-0.5**2)  # 论文经验值
        self.visualGenerate = visualGenerate
        self.T = T
        self.generateGaussianOctaves()  # 在初始化时，顺便构建高斯金字塔(不是差分)

    def generateGaussianOctaves(self):
        tempOctaveImg = self.imgOrigin
        tempOctaveSigma = self.pyramidFirstSigma
        for octaveID in range(self.octaveNumOfPyramid):  # idx小的在下
            tempOctave = self.MyOctave(octaveID, self.sliceNumOfOctave, tempOctaveImg, tempOctaveSigma
                                        , self.sigmaGainPerSlice, self.wantedUsefulDOGs, self.T, self.visualGenerate)  # 会自动调用slice的generate函数
            
            (self.allOctaves).append(tempOctave)  # 加入金字塔

            tempOctaveImg = cv2.pyrDown(tempOctaveImg)  # 为上一层做准备，下采样，缩小为1/2
            tempOctaveSigma = tempOctaveSigma*self.sigmaGainPerOctave


    class MyOctave():
        """
        ClassName       :   
        ClassDescribe   :   内部类
        Specification   :   
        """
        def __init__(self, octaveID, sliceNumOfOctave, octaveOriginImg, octaveFirstSigma, sigmaGainPerSlice, wantedUsefulDOGs, T, visualGenerate = False):
            self.octaveID = octaveID  # octave的编号，小的在下，尺寸跟大
            self.sliceNumOfOctave = sliceNumOfOctave  # octave所含slice数量
            self.DOGNumOfOctave = sliceNumOfOctave -1  # octave所含DOG数量
            self.octaveOriginImg = octaveOriginImg  # 高斯卷积所用的原始图像
            self.slicesShape = octaveOriginImg.shape  # slice尺寸(slice 和 DOG 尺寸相同)
            # [sigma, row, col]
            self.octaveGaussianImgs = np.zeros((self.sliceNumOfOctave,self.slicesShape[0],self.slicesShape[1]),dtype=np.float32)  # 高斯卷积后的各个图像
            # [sigma, row, col]
            self.octaveDOGs = np.zeros((self.DOGNumOfOctave,self.slicesShape[0],self.slicesShape[1]),dtype=np.float32)  # 一层octave差分结果
            self.octaveFirstSigma = octaveFirstSigma  # 初始sigma
            self.octaveSigmas = []  # 高斯卷积的各个sigma
            self.sigmaGainPerSlice = sigmaGainPerSlice  # slice间的sigma放大比例
            self.octaveGaussianKSize = []  # 高斯卷积核的大小
            self.wantedUsefulDOGs = wantedUsefulDOGs
            self.T = T
            self.visualGenerate = visualGenerate
            self.preciseKeyPointsPosInOctave = None
            self.preciseKeyPointsPosInOriginImg = []
            self.generateOctaveSlices()  # 初始化时，顺便将slices生成
            self.generateOctaveDOGs()

        def generateOctaveSlices(self):
            tempSigma = self.octaveFirstSigma
            for sliceID in range(self.sliceNumOfOctave):
                tempKSize = self.get2DGaussianKernelSizeBySigma(tempSigma)
                (self.octaveGaussianKSize).append(tempKSize)

                tempGaussianImg = cv2.GaussianBlur(self.octaveOriginImg, ksize=tempKSize, sigmaX=tempSigma)
                (self.octaveGaussianImgs)[sliceID] = cv2.GaussianBlur(self.octaveOriginImg, ksize=tempKSize, sigmaX=tempSigma)

                (self.octaveSigmas).append(tempSigma)

                if self.visualGenerate:
                    print("[MyOctave] : ", "octaveID : ", self.octaveID, " | slicesShape : "
                            , self.slicesShape, " | sliceID : " ,sliceID ," | kSize : ", tempKSize ," | sigma : ", tempSigma)
                    cv2.imshow("tempGaussianImg", tempGaussianImg)
                    cv2.waitKey(200)
                    cv2.destroyAllWindows()

                tempSigma = tempSigma*self.sigmaGainPerSlice  # 为下一次sigma做准备

        def generateOctaveDOGs(self):
            for tempDOGID in range(self.DOGNumOfOctave):
                tempDOG = self.octaveGaussianImgs[tempDOGID]-self.octaveGaussianImgs[tempDOGID+1]
                (self.octaveDOGs)[tempDOGID] = tempDOG
                if self.visualGenerate:
                    cv2.imshow("tempGaussianImg", tempDOG)
                    cv2.waitKey(200)
                    cv2.destroyAllWindows()

        def get2DGaussianKernelSizeBySigma(self, sigma):  # 使用sigma反推kernelSize，二维元组
            # 利用OpenCV函数： getGaussianKernel()中的ksize推sigma进行反推  # 也有别的反推策略(×3+1、×6+1)
            # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
            # kernelLen = int(((sigma-0.8)/0.3+1)*2+1)  # 利用OpenCV函数反推
            kernelLen = int(3*sigma +1)  # ×3+1的反推策略

            if kernelLen%2.0==0:  # 保证为奇数
                kernelLen = kernelLen+1
            return (kernelLen,kernelLen)

        def locateDiscreteKeyPointsInDOGs(self, wantedUsefulDOGs):
            # 先使用threshold筛选
            contrastThreshold = (self.T)/wantedUsefulDOGs  # 高要求
            # pointThreshold = 0.5*(self.T)/wantedUsefulDOGs  # 低要求
            pointsAfterThresholdInBool = (self.octaveDOGs)>contrastThreshold  # 矩阵操作，返回一个相同形状的布尔值矩阵
            print("[locateDiscreteKeyPointsInDOGs] : No.%d Octave's DOGs' shape"%(self.octaveID),self.octaveDOGs.shape)
            print("[locateDiscreteKeyPointsInDOGs] : No.%d Octave's DOGs 共有 %d 个点超过阈值 %f"%(self.octaveID, pointsAfterThresholdInBool.sum(), contrastThreshold))
            # 找出离散的极值点
            discreteExtremaPointsInBool = np.zeros_like(pointsAfterThresholdInBool, dtype=bool)
            for i in range(1,(self.octaveDOGs).shape[0]-1):  # sigma
                for j in range(1,(self.octaveDOGs).shape[1]-1):  # row
                    for k in range(1,(self.octaveDOGs).shape[2]-1):  # col
                        if pointsAfterThresholdInBool[i,j,k]:  # bool值表示是否通过Threshold检查
                            surroudCube = (self.octaveDOGs)[i-1:i+1,j-1:j+1,k-1:k+1]
                            maxValInCube = surroudCube.max(); minValInCube = surroudCube.min()
                            tempPointVal = (self.octaveDOGs)[i,j,k]  # 当前点的值
                            
                            # if (maxValInCube == tempPointVal and (surroudCube==maxValInCube).sum()==1) \
                            # or (minValInCube == tempPointVal and (surroudCube==minValInCube).sum()==1):  # 极大或极小都算，且周围不允许一样大的或小的
                            if (maxValInCube == tempPointVal) or (minValInCube == tempPointVal):  # 极大或极小都算，且周围允许有一样的
                                discreteExtremaPointsInBool[i,j,k] = True
            print("[locateDiscreteKeyPointsInDOGs] : No.%d DOGs 共有 %d 个离散极值点"%(self.octaveID, discreteExtremaPointsInBool.sum()))
            return discreteExtremaPointsInBool
            
        def locatePreciseKeyPointsInDOGs(self, discreteExtremaPointsInBool, iterateMaxTimes=5, totalOffsetUpperLimit=5, stopOnceOffsetLimit=0.5, derivMatCoeff=1/5):
            """
            找出亚像素极值点
            Related Blogs
            [关键点定位](https://blog.csdn.net/shiyongraow/article/details/78296710)
            [SIFT算法详解](https://blog.csdn.net/zddblog/article/details/7521424)
            [Sift 关键点检测](https://zhuanlan.zhihu.com/p/462061756)
            """
            contrastThreshold = self.T/self.wantedUsefulDOGs
            singularMatrixPoints = 0
            outOfBoundPoints = 0
            noChangeInfeasiblePoints = 0
            keyPointsPosConvertCoeff = 2**(self.octaveID)  # 从Octave中关键点坐标转为原图中的坐标所要乘的系数
            self.preciseKeyPointsPosInOctave = np.zeros_like(discreteExtremaPointsInBool, dtype=bool)  # 和离散一个形状
            for i in range(1,(self.octaveDOGs).shape[0]-1):
                for j in range(1,(self.octaveDOGs).shape[1]-1):
                    for k in range(1,(self.octaveDOGs).shape[2]-1):
                        if discreteExtremaPointsInBool[i,j,k]==True:  # 在离散极值点附近搜索
                                        # 在该点做三元二阶泰勒展开，以获取跟精确的极值点位置
                            pointOrigin = np.matrix([[i],[j],[k]],dtype=np.int32)  # 初始点位置(i,j,k)
                            iterPoint = pointOrigin.copy()  # 记录迭代点的坐标
                            totalOffset = np.zeros_like(pointOrigin,dtype=np.float32)  # 记录迭代的总的偏移量，初始为全零
                            for iter in range(iterateMaxTimes):  # 限制迭代次数
                                derivMatrix, hessianMatrix = self.computeDerivativeAndHessianMatrix(iterPoint, derivMatCoeff)
                                if(np.linalg.det(hessianMatrix)==0):  # 避免后续遇到奇异值矩阵无法迭代
                                    # print("[locatePreciseKeyPointsInDOGs] : Singular Hessian Matrix at point",(i,j,k)," | Iteration : ", iter)
                                    # print(hessianMatrix)
                                    singularMatrixPoints += 1
                                    break
                                tempOffsetFloat = -(hessianMatrix.I)*iterPoint  # 矩阵对象可以通过 .I 更方便的求逆
                                tempOffsetInt = (np.round(tempOffsetFloat)).astype(dtype=np.int32)
                                iterPoint += tempOffsetInt  # 更新当前迭代点的位置

                                # 判断迭代点是否还在范围内
                                posS=iterPoint[0,0]; posX=iterPoint[1,0]; posY=iterPoint[2,0]
                                if (posS<1 or posS>(self.octaveDOGs).shape[0]-2) \
                                or (posX<1 or posX>(self.octaveDOGs).shape[1]-2) \
                                or (posY<1 or posY>(self.octaveDOGs).shape[2]-2):  # 迭代点位置超出范围
                                    outOfBoundPoints += 1
                                    break  # 跳过当前点的后续迭代

                                totalOffset += tempOffsetInt  # 更新总偏移量

                                if (tempOffsetFloat.__abs__()<stopOnceOffsetLimit).sum() == 3:  # 相邻两次迭代的三个分量的偏移都在容许范围内，当前迭代点可能为极值(不一定是(i,j,k))
                                    if (totalOffset.__abs__()<totalOffsetUpperLimit).sum() == 3:  # 和最初的位置的偏移可以接受，符合泰勒展开在初始点附近近似的要求
                                        tempFunctionVal = (self.octaveDOGs)[i,j,k] + 0.5*derivMatrix*tempOffsetFloat  # derivMatrix(1,3)  tempOffsetFloat(3,1)
                                        hessianMatrixXY = hessianMatrix[1:,1:]  # 3×3hessian右下2×2为XY的矩阵
                                        trHessianXY = hessianMatrixXY.trace()
                                        detHessianXY = np.linalg.det(hessianMatrixXY)
                                        # 消除低对比度的点以及边缘效应
                                        if abs(tempFunctionVal) < contrastThreshold:
                                            break
                                        if not ((detHessianXY>0) and ((trHessianXY**2/detHessianXY)<12.1)):
                                            break
                                        # 通过检查
                                        (self.preciseKeyPointsPosInOctave)[posS,posX,posY] = True
                                        (self.preciseKeyPointsPosInOriginImg).append([posS,posX*keyPointsPosConvertCoeff,posY*keyPointsPosConvertCoeff])

                                        # print("[locatePreciseKeyPointsInDOGs] : ",(posS,posX,posY))
                                        break  # 无论如何都break
                                elif (tempOffsetFloat.__abs__()<1).sum() == 3:  # 如果三个坐标变化量不都在容许范围内却又不至于改变当前坐标，则不会收敛了，跳过该点
                                    noChangeInfeasiblePoints += 1
                                    break
            print("[locatePreciseKeyPointsInDOGs] : No.%d DOGs 共有 %d 个亚像素极值点"%(self.octaveID, (self.preciseKeyPointPosInOctave).sum()))
            print("[locatePreciseKeyPointsInDOGs] : singularMatrixPoints    :%d"%(singularMatrixPoints))
            print("[locatePreciseKeyPointsInDOGs] : outOfBoundPoints        :%d"%(outOfBoundPoints))
            print("[locatePreciseKeyPointsInDOGs] : noChangeInfeasiblePoints:%d"%(noChangeInfeasiblePoints))
            print(self.preciseKeyPointsPosInOriginImg)
            


        def computeDerivativeAndHessianMatrix(self, pointCol3D, derivMatCoeff):
            """利用有限差分法求导，pointCol3D是(3*1)列向量，derivMatCoeff用于后续缩放系数"""
            # 一阶偏导和二阶偏导的放缩系数
            firstDerivCoeff = 1/(2*derivMatCoeff)
            secondDerivCoeff = 1/(derivMatCoeff*derivMatCoeff)
            crossDerivCoeff = 1/(4*derivMatCoeff*derivMatCoeff)

            # 提取点坐标[sigma,row,col] 对应 [s,x,y]
            s = pointCol3D[0,0]; x = pointCol3D[1,0]; y = pointCol3D[2,0]
            # 一阶偏导
            ds = (self.octaveDOGs[s+1,x,y]-self.octaveDOGs[s-1,x,y])*firstDerivCoeff  # 同一层octave，s越大sigma越大
            dx = (self.octaveDOGs[s,x+1,y]-self.octaveDOGs[s,x-1,y])*firstDerivCoeff
            dy = (self.octaveDOGs[s,x,y+1]-self.octaveDOGs[s,x,y-1])*firstDerivCoeff
            # 二阶偏导
            dss = (self.octaveDOGs[s+1,x,y]+self.octaveDOGs[s-1,x,y]-2*self.octaveDOGs[s,x,y])*secondDerivCoeff
            dxx = (self.octaveDOGs[s,x+1,y]+self.octaveDOGs[s,x-1,y]-2*self.octaveDOGs[s,x,y])*secondDerivCoeff
            dyy = (self.octaveDOGs[s,x,y+1]+self.octaveDOGs[s,x,y-1]-2*self.octaveDOGs[s,x,y])*secondDerivCoeff
            # 交叉偏导
            dsx = ((self.octaveDOGs[s+1,x+1,y]+self.octaveDOGs[s-1,x-1,y])-(self.octaveDOGs[s+1,x-1,y]+self.octaveDOGs[s-1,x+1,y]))*crossDerivCoeff
            dxs = dsx
            dxy = ((self.octaveDOGs[s,x+1,y+1]+self.octaveDOGs[s,x-1,y-1])-(self.octaveDOGs[s,x+1,y-1]+self.octaveDOGs[s,x-1,y+1]))*crossDerivCoeff
            dyx = dxy
            dys = ((self.octaveDOGs[s+1,x,y+1]+self.octaveDOGs[s-1,x,y-1])-(self.octaveDOGs[s+1,x,y-1]+self.octaveDOGs[s-1,x,y+1]))*crossDerivCoeff
            dsy = dys
            # 矩阵构建
            DerivMat1by3 = np.matrix([ds,dx,dy], dtype=np.float32)
            HessianMat3by3 = np.matrix([[dss,dsx,dsy],
                                        [dxs,dxx,dxy],
                                        [dys,dyx,dyy]],dtype=np.float32)
            # 返回值
            return DerivMat1by3, HessianMat3by3

        def showPreciseKeyPointsInOriginImg(self, imgOrigin):
            pass


def MySIFT(img):
    """
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①②③④⑤⑥⑦⑧⑨⑩
    Specification   :   Related Blog : https://www.bilibili.com/video/BV1Qb411W7cK/

    Related Blogs   :
    [关键点定位]:https://blog.csdn.net/shiyongraow/article/details/78296710)
    [SIFT算法详解](https://blog.csdn.net/zddblog/article/details/7521424)
    [Sift 关键点检测](https://zhuanlan.zhihu.com/p/462061756)

    """

    pass




if __name__ == "__main__":
    img = cv2.imread(r"./PicsForCode/FeatureExtract/fdfz02.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pyramid = MyDifferenceOfGaussianPyramid(img,None,2,visualGenerate=True)
    discreteExtremaPointsInBool = (pyramid.allOctaves[0]).locateDiscreteKeyPointsInDOGs(pyramid.wantedUsefulDOGs)
    preciseExtremaPointsInBool = (pyramid.allOctaves[0]).locatePreciseKeyPointsInDOGs(discreteExtremaPointsInBool)



    pass
