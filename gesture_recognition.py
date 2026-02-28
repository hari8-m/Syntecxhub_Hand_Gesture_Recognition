import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('/content/hand.jpg')

if img is None:
    print("Image not found")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    _, thresh = cv2.threshold(
        blur, 120, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw hand contour
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
