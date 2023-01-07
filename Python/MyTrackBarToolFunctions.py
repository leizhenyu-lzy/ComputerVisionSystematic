import cv2


def cannyWithTrackBar(outerPath, onChangeCannyThreshLow=None, onChangeCannyThreshHigh=None):
    r"""
    FunctionName    : myCannyWithTrackBar
    FunctionDescribe: 使用OpenCV2的Canny结合Trackbar，用于选出Canny的两个阈值，按下'q'键退出
    InputParameter  : ①outerPath(str 图片路径)  
                      ②onChangeCannyThreshLow(默认None)  
                      ③onChangeCannyThreshHigh(默认None)
    OutputParameter : ①imgCanny(<class 'numpy.ndarray'>)  
                      ②cannyThresholdLow  
                      ③cannyThresholdHigh
    """
    path = outerPath
    imgGray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 对应两个trackbar的onChange函数
    def recallCannyThresholdLow(threshold):
        # print("recallCannyThresholdLow: ", threshold)
        pass
    def recallCannyThresholdHigh(threshold):
        # print("recallCannyThresholdLow: ", threshold)
        pass

    cv2.namedWindow("CannyThresholds")
    cv2.resizeWindow("CannyThresholds", 500, 100)
    if (onChangeCannyThreshHigh is None) or (onChangeCannyThreshLow is None):
        cv2.createTrackbar("CannyThreshLow","CannyThresholds",0,255,recallCannyThresholdLow)
        cv2.createTrackbar("CannyThreshHigh","CannyThresholds",0,255,recallCannyThresholdHigh)
    else:
        cv2.createTrackbar("CannyThreshLow","CannyThresholds",0,255,onChangeCannyThreshLow)
        cv2.createTrackbar("CannyThreshHigh","CannyThresholds",0,255,onChangeCannyThreshHigh)

    while True:
        thresholdLow = cv2.getTrackbarPos("CannyThreshLow", "CannyThresholds")
        thresholdHigh = cv2.getTrackbarPos("CannyThreshHigh", "CannyThresholds")
        imgCanny = cv2.Canny(imgGray,thresholdLow,thresholdHigh)
        cv2.imshow("imgCanny", imgCanny)
        
        if cv2.waitKey(1)&0xFF == ord('q'):
            cv2.destroyAllWindows()
            print("CannyThresholdLow: ",min(thresholdHigh,thresholdLow),
                    "\nCannyThresholdHigh: ",max(thresholdHigh,thresholdLow))
            print(type(imgCanny))
            return imgCanny, min(thresholdHigh,thresholdLow), max(thresholdHigh,thresholdLow)


if __name__ == "__main__":
    cannyWithTrackBar("./PicsForCode/StraightLines/StraightLines01.jpg")


    