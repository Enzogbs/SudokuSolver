import numpy as np
import matplotlib.pyplot as plt
from utils import *
from Solver import *
import cv2

# PREPARE THE IMAGE
img = cv2.imread("img2.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1000, 1000))
img = cv2.GaussianBlur(img, (5, 5), 1)
th2 = cv2.adaptiveThreshold(img,255,1, 1,11,2)

# FIND CONTOURS
imgContours = img.copy()
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 3)
biggest, max_area = biggest_contour(contours)

# PERSPECTIVE TRANSFORM
input_pts = reorder_vertices(biggest)
maxHeight, maxWidth = output_dimensions(*input_pts)
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

imgBigContours = img.copy()
cv2.drawContours(imgBigContours, biggest, -1, (255, 0, 0), 5)
Matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
ImgWarp = cv2.warpPerspective(imgContours, Matrix,(maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
ImgWarp = cv2.resize(ImgWarp, (450, 450))
ImgWarp_processed = process_warped_image(ImgWarp)

# SPLIT NUMBER BOXES
boxes = SplitBoxes(ImgWarp_processed)
numbers = list(map(extract, boxes))
numbers = np.array(numbers).reshape(9,9)
grid = numbers.copy()

# SOLVE SUDOKU
if (Suduko(grid, 0, 0)):
    print(grid)
answers = np.array(grid) - numbers

# PREPARE ANSWERS MASK
mask = get_answer_grid(answers)
mask = cv2.resize(mask, (maxWidth, maxHeight))

# UNWARP SOLUTION GRID
input_pts = np.float32([[0, 0],
                        [0, maxHeight],
                        [maxWidth, maxHeight],
                        [maxWidth, 0]])

output_pts = reorder_vertices(biggest)
Matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
UnWarpedSolution = cv2.warpPerspective(mask, Matrix,(img.shape[0],img.shape[1]), flags=cv2.INTER_LINEAR)

# MERGE UNWARPED SOLUTION GRID WITH INICIAL IMAGE
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
solution = cv2.bitwise_or(img.astype("int64"), UnWarpedSolution.astype("int64"))

# DISPLAY IMAGE
plt.imshow(solution, cmap="gray")
plt.show()
