import cv2 as cv

# https://github.com/aleker/PyHanafudaClassifier
# ^^^ check out this github
# helpful video: https://www.youtube.com/watch?v=m-QPjO-2IkA
# https://towardsdatascience.com/computer-vision-for-beginners-part-2-29b3f9151874


def detect(image):
    # grayscale
    # blurred
    # thresholded
    # contour detection
    # find number of cards

    image = cv.imread(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.blur(gray, (10, 10))
    #blurred = cv.medianBlur(gray, 21)
    _, threshold = cv.threshold(blurred, 100, 225, cv.THRESH_BINARY_INV)
    #threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    approx = []
    for contour in contours:
        epsilon = 0.1 * cv.arcLength(contour, True)
        approx.append(cv.approxPolyDP(contour, epsilon, True))

    # Showing images created
    #cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.drawContours(image, approx, -1, (0, 255, 0), 3)
    cv.imshow('Original image', image)
    #cv.imshow('Gray image', gray)
    cv.imshow('Blurred image', blurred)
    cv.imshow("Thresholded Image", threshold)
    #cv.imshow("Adaptive Thresholded Image", threshold2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 1


def identify(image):
    # approximate corner point of each card
    # transform card into flattened perspective image
    # ? create color masks of each card (query image)
    # ? compare mask to train images
    return 1


detect("images/IMG_8280.jpg")
