import cv2
import operator
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 28
RESIZED_IMAGE_HEIGHT = 28


class ContourWithData:
    npaContour = None  # contour
    boundingRect = None  # bounding rect for contour
    intRectX = 0  # bounding rect top left corner x location
    intRectY = 0  # bounding rect top left corner y location
    intRectWidth = 0  # bounding rect width
    intRectHeight = 0  # bounding rect height
    fltArea = 0.0  # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def IsContourValid(self):
        if self.fltArea < MIN_CONTOUR_AREA:
            return False
        return True


def main():
    allContoursWithData = []
    validContoursWithData = []
    imgTestingNumbers = cv2.imread("TEST.jpg")

    if imgTestingNumbers is None:
        print("error: Cannot read image from!! \n\n")
        os.system("pause")
        return

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    imgThreshCopy = imgThresh.copy()
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.IsContourValid():
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key=operator.attrgetter("intRectX"))

    for contourWithData in validContoursWithData:
        cv2.rectangle(imgTestingNumbers,
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),
                      (0, 255, 0),
                      4)

    cv2.imshow("imgTestingNumbers", cv2.resize(imgTestingNumbers, (700, 700)))
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
# end if
