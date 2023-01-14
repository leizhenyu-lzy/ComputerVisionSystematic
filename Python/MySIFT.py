# Scale-Invariant Feature Transform
import cv2
import numpy as np
import math

class MyDifferenceOfGaussianPyramid():
    """
    ClassName       :   高斯差分金字塔 \n
    ClassDescribe   :   \n
    Specification   :   \n`
    """

    def __init__(self, imgGray, octaveNumOfPyramid = None, wantedUsefulDOGs = None, T=0.04, visualGenerateImshowTime=-1):
        """
        FunctionDescribe:   
                            Init the DOGs Pyramid. Octaves模拟近大远小，高斯核卷积模拟清晰和模糊 \n
        InputParameter  :   
                            ①imgGray只接受单通道的灰度图，且数据为整形[0,255] \n
                            ②octaveNumOfPyramid(>0) : 如果是None，则使用论文推荐值 \n
                            ③wantedUsefulDOGs(>0)   : 一层octave差分得到的DOGs中可用于三维特征提取的层数，其值+3即为一层octave所含的slices数量 \n 
                                                        +3的解释：首先DOGs的上下两层不能使用(无法求导找极值点)，所以+2;其次是差分所得，所以要再+1 \n
                            ④T
        OutputParameter :   
                            ①②③④⑤⑥⑦⑧⑨⑩ \n
        Specification   :   
                            Related Blog : https://www.bilibili.com/video/BV1Qb411W7cK/ \n
        """
        if len(imgGray.shape)!=2:  # 图片维数必须为2
            print("[myDifferenceOfGaussianPyramid::__init__] : The img's dimension is not 2.")
            exit(-1)
        self.imgGray = imgGray

        if octaveNumOfPyramid is None :
            self.octaveNumOfPyramid = int(math.log2(min(imgGray.shape[0], imgGray.shape[1])))-3  # 金字塔大层数（论文推荐数量）
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
        self.pyramidFirstSigma = math.sqrt(1.6**2-0.5**2)  # 论文经验值  # 0.5是摄像图初始的模糊度，1.6是论文认为的需要达到的模糊度
        self.visualGenerateImshowTime = visualGenerateImshowTime
        self.T = T
        self.contrastThreshold = (self.T)/(self.wantedUsefulDOGs)*255  # 乘以255相当于重映射[0,1]->[0,255]
        self.generateGaussianOctaves()  # 在初始化时，顺便构建高斯金字塔(不是差分)

    def generateGaussianOctaves(self):
        tempOctaveImg = self.imgGray
        tempOctaveSigma = self.pyramidFirstSigma
        for octaveID in range(self.octaveNumOfPyramid):  # idx小的在下
            tempOctave = self.MyOctave(octaveID, self.sliceNumOfOctave, tempOctaveImg, tempOctaveSigma
                                        , self.sigmaGainPerSlice, self.contrastThreshold, self.visualGenerateImshowTime)  # 会自动调用slice的generate函数
            
            (self.allOctaves).append(tempOctave)  # 加入金字塔

            tempOctaveImg = cv2.pyrDown(tempOctaveImg)  # 为上一层做准备，下采样，缩小为1/2
            tempOctaveSigma = tempOctaveSigma*self.sigmaGainPerOctave
    
    def locatePreciseKeyPointsInAllOcataves(self, iterateMaxTimes=5, totalOffsetUpperLimit=5, stopOnceOffsetLimit=0.5, derivMatCoeff=1):
        """
        iterateMaxTimes=5 最大迭代搜索次数 \n
        totalOffsetUpperLimit=5 最大总偏移距离 \n
        stopOnceOffsetLimit=0.5 停止迭代的单次三维 \n
        derivMatCoeff=1 求黑赛矩阵和偏导行向量的附加系数
        """
        for octave in self.allOctaves:
            octave.locatePreciseKeyPointsInDOGs(iterateMaxTimes, totalOffsetUpperLimit, stopOnceOffsetLimit, derivMatCoeff)

    def showPreciseKeyPointsInAllOctaves(self, outerImg=None):
        for octave in self.allOctaves:
            octave.showPreciseKeyPoints(outerImg)


    class MyOctave():
        """
        ClassName       :   
        ClassDescribe   :   内部类
        Specification   :   
        """
        def __init__(self, octaveID, sliceNumOfOctave, octaveGrayImg, octaveFirstSigma, sigmaGainPerSlice, 
                        contrastThreshold, visualGenerateImshowTime=-1):
            self.octaveID = octaveID  # octave的编号，小的在下，尺寸跟大
            self.sliceNumOfOctave = sliceNumOfOctave  # octave所含slice数量
            self.DOGNumOfOctave = sliceNumOfOctave -1  # octave所含DOG数量
            self.octaveGrayImg = octaveGrayImg  # 高斯卷积所用的原始图像
            self.slicesShape = octaveGrayImg.shape  # slice尺寸(slice 和 DOG 尺寸相同)
            # [sigma, row, col]
            self.octaveGaussianImgs = np.zeros((self.sliceNumOfOctave,self.slicesShape[0],self.slicesShape[1]),dtype=np.float32)  # 高斯卷积后的各个图像
            # [sigma, row, col]
            self.octaveDOGs = np.zeros((self.DOGNumOfOctave,self.slicesShape[0],self.slicesShape[1]),dtype=np.float32)  # 一层octave差分结果
            self.octaveFirstSigma = octaveFirstSigma  # 初始sigma
            self.octaveSigmas = []  # 高斯卷积的各个sigma
            self.sigmaGainPerSlice = sigmaGainPerSlice  # slice间的sigma放大比例
            self.octaveGaussianKSize = []  # 高斯卷积核的大小
            self.contrastThreshold = contrastThreshold
            self.visualGenerateImshowTime = visualGenerateImshowTime
            self.preciseKeyPointsPosInOctaveList = []
            self.preciseKeyPointsPosInOriginImgList = []
            self.generateOctaveSlices()  # 初始化时，顺便将slices生成
            self.generateOctaveDOGs()
            # print(self.octaveSigmas)
            # exit(-1)

        def generateOctaveSlices(self):
            tempSigma = self.octaveFirstSigma
            for sliceID in range(self.sliceNumOfOctave):
                tempKSize = self.get2DGaussianKernelSizeBySigma(tempSigma)
                (self.octaveGaussianKSize).append(tempKSize)

                tempGaussianImg = cv2.GaussianBlur(self.octaveGrayImg, ksize=tempKSize, sigmaX=tempSigma)
                (self.octaveGaussianImgs)[sliceID] = cv2.GaussianBlur(self.octaveGrayImg, ksize=tempKSize, sigmaX=tempSigma)

                (self.octaveSigmas).append(tempSigma)

                # if (self.visualGenerateImshowTime) >= 0:
                #     print("[MyOctave] : ", "octaveID : ", self.octaveID, " | slicesShape : "
                #             , self.slicesShape, " | sliceID : " ,sliceID ," | kSize : ", tempKSize ," | sigma : ", tempSigma)
                #     cv2.imshow("tempGaussianImg", tempGaussianImg)
                #     cv2.waitKey(self.visualGenerateImshowTime)
                #     cv2.destroyAllWindows()

                tempSigma = tempSigma*self.sigmaGainPerSlice  # 为下一次sigma做准备

        def generateOctaveDOGs(self):
            for imgIdx, tempDOGID in enumerate(range(self.DOGNumOfOctave)):
                tempDOG = self.octaveGaussianImgs[tempDOGID]-self.octaveGaussianImgs[tempDOGID+1]
                (self.octaveDOGs)[tempDOGID] = tempDOG
                if (self.visualGenerateImshowTime) >= 0:
                    cv2.imshow("Octave ID: %d | DOG No: %d"%(self.octaveID, imgIdx), tempDOG)
                    cv2.waitKey(self.visualGenerateImshowTime)
                    cv2.destroyAllWindows()

        def get2DGaussianKernelSizeBySigma(self, sigma):  # 使用sigma反推kernelSize，二维元组
            # 利用OpenCV函数： getGaussianKernel()中的ksize推sigma进行反推  # 也有别的反推策略(×3+1、×6+1)
            # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
            # kernelLen = int(((sigma-0.8)/0.3+1)*2+1)  # 利用OpenCV函数反推
            kernelLen = int(3*sigma +1)  # ×3+1的反推策略
            kernelLen = int(6*sigma +1)  # ×6+1的反推策略

            if kernelLen%2.0==0:  # 保证为奇数
                kernelLen = kernelLen+1
            return (kernelLen,kernelLen)

        def locateDiscreteKeyPointsInDOGs(self):
            """搜索离散极值点，粗糙的非极大值抑制"""
            # 先使用threshold筛选
            pointsAfterThresholdInBool = (self.octaveDOGs)>self.contrastThreshold  # 矩阵操作，返回一个相同形状的布尔值矩阵
            print("[locateDiscreteKeyPointsInDOGs] : No.%d Octave's DOGs' shape"%(self.octaveID),self.octaveDOGs.shape)
            print("[locateDiscreteKeyPointsInDOGs] : No.%d Octave's DOGs 共有 %d 个点超过阈值 %f"%(self.octaveID, pointsAfterThresholdInBool.sum(), self.contrastThreshold))
            # 找出离散的极值点
            discreteExtremaPointsInBool = np.zeros_like(pointsAfterThresholdInBool, dtype=bool)
            for i in range(1,(self.octaveDOGs).shape[0]-1):  # sigma
                for j in range(1,(self.octaveDOGs).shape[1]-1):  # row
                    for k in range(1,(self.octaveDOGs).shape[2]-1):  # col
                        if pointsAfterThresholdInBool[i,j,k]:  # bool值表示是否通过Threshold检查
                            surroudCube = (self.octaveDOGs)[i-1:i+1,j-1:j+1,k-1:k+1]
                            maxValInCube = surroudCube.max(); minValInCube = surroudCube.min()
                            tempPointVal = (self.octaveDOGs)[i,j,k]  # 当前点的值
                            
                            if (maxValInCube == tempPointVal and (surroudCube==maxValInCube).sum()==1) \
                            or (minValInCube == tempPointVal and (surroudCube==minValInCube).sum()==1):  # 极大或极小都算，且周围不允许一样大的或小的
                            # if (maxValInCube == tempPointVal) or (minValInCube == tempPointVal):  # 极大或极小都算，且周围允许有一样的
                                discreteExtremaPointsInBool[i,j,k] = True
            print("[locateDiscreteKeyPointsInDOGs] : No.%d DOGs 共有 %d 个离散极值点"%(self.octaveID, discreteExtremaPointsInBool.sum()))
            return discreteExtremaPointsInBool
            
        def locatePreciseKeyPointsInDOGs(self, iterateMaxTimes=5, totalOffsetUpperLimit=5, stopOnceOffsetLimit=0.5, derivMatCoeff=1):
            """
            找出亚像素极值点，内部先调用离散搜索　\n
            Related Blogs \n
            [关键点定位](https://blog.csdn.net/shiyongraow/article/details/78296710) \n
            [SIFT算法详解](https://blog.csdn.net/zddblog/article/details/7521424) \n
            [Sift 关键点检测](https://zhuanlan.zhihu.com/p/462061756)
            """
            discreteExtremaPointsInBool = self.locateDiscreteKeyPointsInDOGs()
            # singularMatrixPoints = 0
            # outOfBoundPoints = 0
            # noChangeInfeasiblePoints = 0
            keyPointsPosConvertCoeff = 2**(self.octaveID)  # 从Octave中关键点坐标转为原图中的坐标所要乘的系数
            for i in range(1,(self.octaveDOGs).shape[0]-1):
                for j in range(1,(self.octaveDOGs).shape[1]-1):
                    for k in range(1,(self.octaveDOGs).shape[2]-1):
                        if discreteExtremaPointsInBool[i,j,k]==True:  # 在离散极值点附近搜索
                                        # 在该点做三元二阶泰勒展开，以获取跟精确的极值点位置
                            pointOrigin = np.matrix([[i],[j],[k]],dtype=np.int32)  # 初始点位置(i,j,k)
                            iterPoint = pointOrigin.copy()  # 记录迭代点的坐标
                            totalOffset = np.zeros_like(pointOrigin,dtype=np.float32)  # 记录迭代的总的偏移量，初始为全零
                            for iter in range(iterateMaxTimes):  # 限制迭代次数
                                derivRowVector, hessianMatrix = self.computeDerivativeAndHessianMatrix(iterPoint, derivMatCoeff)
                                # 不使用伪逆，需要跳过奇异矩阵，避免后续遇到奇异值矩阵无法迭代
                                # if(np.linalg.det(hessianMatrix)==0):
                                #     # print("[locatePreciseKeyPointsInDOGs] : Singular Hessian Matrix at point",(i,j,k)," | Iteration : ", iter)
                                #     # print(hessianMatrix)
                                #     singularMatrixPoints += 1
                                #     break
                                # hessianMatrixInv = np.linalg.inv(hessianMatrix)
                                # 使用伪逆
                                hessianMatrixInv = np.linalg.pinv(hessianMatrix)

                                tempOffsetFloat = -hessianMatrixInv*(derivRowVector.T)  # 矩阵对象可以通过 .I 更方便的求逆
                                tempOffsetInt = (np.round(tempOffsetFloat)).astype(dtype=np.int32)
                                iterPoint += tempOffsetInt  # 更新当前迭代点的位置

                                # 判断迭代点是否还在范围内
                                posS=iterPoint[0,0]; posX=iterPoint[1,0]; posY=iterPoint[2,0]
                                if (posS<1 or posS>(self.octaveDOGs).shape[0]-2) \
                                or (posX<1 or posX>(self.octaveDOGs).shape[1]-2) \
                                or (posY<1 or posY>(self.octaveDOGs).shape[2]-2):  # 迭代点位置超出范围
                                    # outOfBoundPoints += 1
                                    break  # 跳过当前点的后续迭代

                                totalOffset += tempOffsetInt  # 更新总偏移量

                                if (tempOffsetFloat.__abs__()<stopOnceOffsetLimit).sum() == 3:  # 相邻两次迭代的三个分量的偏移都在容许范围内，当前迭代点可能为极值(不一定是(i,j,k))
                                    if (totalOffset.__abs__()<totalOffsetUpperLimit).sum() == 3:  # 和最初的位置的偏移可以接受，符合泰勒展开在初始点附近近似的要求
                                        tempFunctionVal = (self.octaveDOGs)[i,j,k] + 0.5*derivRowVector*tempOffsetFloat  # derivMatrix(1,3)  tempOffsetFloat(3,1)
                                        hessianMatrixXY = hessianMatrix[1:,1:]  # 3×3hessian右下2×2为XY的矩阵
                                        trHessianXY = hessianMatrixXY.trace()
                                        detHessianXY = np.linalg.det(hessianMatrixXY)
                                        # 消除低对比度的点以及边缘效应
                                        if abs(tempFunctionVal) < self.contrastThreshold:
                                            break
                                        if not ((detHessianXY>0) and ((trHessianXY**2/detHessianXY)<12.1)):
                                            break
                                        # 通过检查
                                        if [posS,posX,posY] not in self.preciseKeyPointsPosInOctaveList:  # 防止重复
                                            (self.preciseKeyPointsPosInOctaveList).append([posS,posX,posY])
                                            (self.preciseKeyPointsPosInOriginImgList).append([posS,posX*keyPointsPosConvertCoeff,posY*keyPointsPosConvertCoeff])

                                        # print("[locatePreciseKeyPointsInDOGs] : ",(posS,posX,posY))
                                        break  # 无论如何都break
                                elif (tempOffsetFloat.__abs__()<1).sum() == 3:  # 如果三个坐标变化量不都在容许范围内却又不至于改变当前坐标，则不会收敛了，跳过该点
                                    # noChangeInfeasiblePoints += 1
                                    break
            print("[locatePreciseKeyPointsInDOGs] : No.%d DOGs 共有 %d 个亚像素极值点"%(self.octaveID, len(self.preciseKeyPointsPosInOctaveList)))
            # print("[locatePreciseKeyPointsInDOGs] : singularMatrixPoints    :%d"%(singularMatrixPoints))
            # print("[locatePreciseKeyPointsInDOGs] : outOfBoundPoints        :%d"%(outOfBoundPoints))
            # print("[locatePreciseKeyPointsInDOGs] : noChangeInfeasiblePoints:%d"%(noChangeInfeasiblePoints))
            print("[locatePreciseKeyPointsInDOGs] : 在 OctaveID=%d 找到 %d 个精确的极值点"%(self.octaveID, len(self.preciseKeyPointsPosInOctaveList)))

        def computeDerivativeAndHessianMatrix(self, pointCol3D, derivMatCoeff):
            """利用有限差分法求导，返回，　\n
            输入的pointCol3D是(3*1)列向量，derivMatCoeff是用于后续缩放系数"""
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

        def showPreciseKeyPoints(self, outerImg=None):
            """都是使用图片的拷贝，不必担心改变原图"""
            if outerImg is None:  # 没有传入外部图片,则在缩小的图片中展示关键点
                imgWithKeyPoints = ((self.octaveGrayImg).copy()).astype(np.uint8)  # 转为uint8否则imshow显示不出正确的灰度图
                if len(self.preciseKeyPointsPosInOctaveList)==0:
                    return
                for pos in self.preciseKeyPointsPosInOctaveList:
                    # 注意圆心坐标不是row-col坐标系，而是x-y坐标系
                    cv2.circle(imgWithKeyPoints,center=(pos[2],pos[1]),radius=pos[0]*4,color=0,thickness=1)  # 灰度图，故color使用0，而非元组
                cv2.imshow("ImgWithKeyPoints OctaveID: %d"%(self.octaveID), imgWithKeyPoints)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:  # 传入外部图片，展示完整尺寸
                imgWithKeyPoints = outerImg.copy()
                if len(self.preciseKeyPointsPosInOriginImgList)==0:
                    return
                for pos in self.preciseKeyPointsPosInOriginImgList:
                    # 注意圆心坐标不是row-col坐标系，而是x-y坐标系
                    cv2.circle(imgWithKeyPoints,center=(pos[2],pos[1]),radius=pos[0]*4,color=(0,255,0),thickness=1)
                cv2.imshow("ImgWithKeyPoints OctaveID: %d"%(self.octaveID), imgWithKeyPoints)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


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
    imgOrigin = cv2.imread(r"./PicsForCode/FeatureExtract/wuke02.jpeg")
    imgGray = cv2.imread(r"./PicsForCode/FeatureExtract/wuke02.jpeg", cv2.IMREAD_GRAYSCALE)
    imgMean = imgOrigin.mean(axis=-1).astype(np.uint8)

    imgUse = imgMean

    imgUse = cv2.pyrDown(imgUse)
    imgUse = cv2.pyrDown(imgUse)
    imgUse = cv2.pyrDown(imgUse)
    imgUse = cv2.pyrUp(imgUse)
    imgUse = cv2.pyrUp(imgUse)
    imgUse = cv2.pyrUp(imgUse)

    pyramid = MyDifferenceOfGaussianPyramid(imgUse,None,2,visualGenerateImshowTime=0)
    pyramid.locatePreciseKeyPointsInAllOcataves()
    pyramid.showPreciseKeyPointsInAllOctaves()

    pass
