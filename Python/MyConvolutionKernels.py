import cv2 as cv
import numpy as np
import math

def myDiscreteLaplacianRough3By3(zoomCoeff=1):
    r"""TODO
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   
    """
    return np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.int8)/(zoomCoeff**2)

def myDiscreteLaplacianAccurate3By3(zoomCoeff=1):
    r"""TODO
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   
    """
    return np.array([[1,4,1],[4,-20,4],[1,4,1]],dtype=np.int8)/(zoomCoeff**2)
    
def myGaussian1DFunction(sigma=1,xPos=0):
    r"""
    FunctionName    :   myGaussian1DFunction
    FunctionDescribe:   
    InputParameter  :   ①sigma(标准差，不是方差)
                        ②xPos(横坐标)
    OutputParameter :   ①
    """
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-xPos**2/(2*sigma**2))

def myGaussian1DKernel(kernelLength=3,sigma=1):
    r"""
    FunctionName    :   myGaussian1DKernel
    FunctionDescribe:   求出一维高斯核（行向量）
    InputParameter  :   ①kernelLength(一维的长度，需要是正奇数，否则return None)
                        ②sigma(标准差，不是方差)
    OutputParameter :   ①kernel(列向量，<class 'numpy.ndarray'>)
    OfficialLink    :   https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    Specification   :   和官方的结果略有不同，小数点后第8位存在一定的误差
    """
    if(kernelLength%2 != 1) or (kernelLength<=0):
        print("Kernel Length Should Be Odd and Positive.")
        return None

    kernel = np.zeros((1,kernelLength),dtype=np.float32)
    # print(kernel.shape)
    midPos = int((kernelLength-1)*0.5)
    kernelSum = 0
    for i in range(0,midPos+1):
        tempVal = math.exp(-(i-midPos)**2/(2*sigma**2))
        kernel[0,i] = tempVal
        kernel[0,kernelLength-i-1] = tempVal  # 轴对称
        kernelSum+=tempVal*2
    kernelSum -= kernel[0,midPos]  # 中间位置算了两遍，减掉一遍
    normalizedKernel = kernel/kernelSum  # normalization
    return normalizedKernel

def myGaussian2DFunction(sigma=1,xyPos=(0,0)):
    r"""
    FunctionName    :   myGaussian2DFunction
    FunctionDescribe:   自己写的二维高斯函数
    InputParameter  :   ①sigma(标准差，不是方差)
                        ②xyPos(待求点处的函数值)
    OutputParameter :   ①result(函数值)
    """
    if len(xyPos)!=2:
        return None
    xPos = xyPos[0]; yPos = xyPos[1]
    variance = sigma**2
    result = myGaussian1DFunction(sigma,xPos)*myGaussian1DFunction(sigma,yPos)  # 不相关&相互独立的二维正态分布，概率密度直接相乘
    # resultAnother = (1/(2*math.pi*variance))*math.exp(-(xPos**2+yPos**2)/(2*variance))  # 用二维正态分布公式
    # print("Diff of 2 calculating methods: ", result - resultAnother)  # 检查两种计算方式的差值
    return result

def myGaussian2DKernel(kernelSize=3,sigma=1):
    r"""
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   kernel元素之和不一定为1，可能会有些误差。eg：myGaussian2DKernel(5,5)的和为0.99999994
    """
    gaussian1DKernel = myGaussian1DKernel(kernelSize,sigma)  # 行向量 shape:(1,x)
    return gaussian1DKernel.T*gaussian1DKernel

def myLaplacianOfGaussianFunction(sigma=1, xyPos=(0,0)):
    r"""
    FunctionName    :   
    FunctionDescribe:   LOG
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   倒置墨西哥帽
    """
    if len(xyPos)!=2:
        return None
    xPos = xyPos[0]; yPos = xyPos[1]
    # 两种计算方式，上面的是直接利用公式，下面是利用LOG和二维Gaussian的关系
    return (xPos**2+yPos**2-2*sigma**2)*math.exp(-(xPos**2+yPos**2)/(2*sigma**2))/(2*math.pi*(sigma**6))
    # return (xPos**2+yPos**2-2*sigma**2)/(sigma**4)*myGaussian2DFunction(sigma, xyPos)

def myLaplacianOfGaussianKernel(kernelSize=3,sigma=1,zoomCoeff=(2*math.pi)):
    r"""
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   检查两侧LOG响应值符号是否相反，如果一正一负，说明中间存在灰度剧烈变化的点，可以看作边缘
                        两侧的平坦区域，LOG 响应值 = 0；噪声处于 LOG > 0 区域时，LOG 响应值 > 0；噪声处于 LOG < 0 区域时，LOG 响应值 < 0
                        无法完全去除噪声
    """
    if (kernelSize<=0) or (kernelSize%2!=1):
        return None
    kernel = np.zeros((kernelSize,kernelSize),dtype=np.float32)
    # 中心对称+旋转对称，仅求1/8个矩阵即可
    kernelHalfSize = int((kernelSize-1)/2)
    centerVal = myLaplacianOfGaussianFunction(sigma, (0,0))*zoomCoeff
    kernel[kernelHalfSize,kernelHalfSize] = centerVal  # 提前算好中心点
    print("centerVal: ",centerVal)
    # 算-90～-45的矩阵，并利用坐标平移
    for i in range(1,kernelHalfSize+1):
        for j in range(0,i+1):
            curVal = myLaplacianOfGaussianFunction(sigma, (i,j))*zoomCoeff
            print(i,j,curVal)
            if j!=0:  # 不在十字轴线上，算8次
                kernel[kernelHalfSize+i,kernelHalfSize+j] = curVal; kernel[kernelHalfSize+j,kernelHalfSize+i] = curVal
                kernel[kernelHalfSize-i,kernelHalfSize+j] = curVal; kernel[kernelHalfSize+j,kernelHalfSize-i] = curVal
                kernel[kernelHalfSize+i,kernelHalfSize-j] = curVal; kernel[kernelHalfSize-j,kernelHalfSize+i] = curVal
                kernel[kernelHalfSize-i,kernelHalfSize-j] = curVal; kernel[kernelHalfSize-j,kernelHalfSize-i] = curVal
            else:  # 在十字轴线上，算4次
                kernel[kernelHalfSize+i,kernelHalfSize] = curVal; kernel[kernelHalfSize-i,kernelHalfSize] = curVal
                kernel[kernelHalfSize,kernelHalfSize+i] = curVal; kernel[kernelHalfSize,kernelHalfSize-i] = curVal
    # print(kernel)
    kernel = kernel-(np.sum(kernel))/(kernelSize**2)  # 使得矩阵和为0
    # kernel[kernelHalfSize,kernelHalfSize] -= np.sum(kernel)  # 再次精细化
    print("[myLaplacianOfGaussianKernel]:   sum(LOG Kernel): ", np.sum(kernel))
    return kernel


if __name__ == "__main__":
    r"""
    FunctionName    :   
    FunctionDescribe:   
    InputParameter  :   ①
    OutputParameter :   ①
    Specification   :   
    """
    sigma = 1
    print(myLaplacianOfGaussianFunction(sigma, (1,0)))
    print(myLaplacianOfGaussianKernel(7,sigma))


    pass