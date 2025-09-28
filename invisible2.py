import cv2
import numpy as np
import time

video = cv2.VideoCapture(0)
time.sleep(3)

# Capture background
background = 0
for i in range(60):   # capture more frames for a stable background
    ret, background = video.read()
background = np.flip(background, axis=1)

while True:
    ret, image = video.read()
    if not ret:
        break

    image = np.flip(image, axis=1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Red ranges in HSV ---
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both masks
    mask = mask1 + mask2

    # Clean mask: remove noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask = mask.astype(np.uint8)  # Ensure mask is uint8 for OpenCV functions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Invert mask to get non-red areas
    mask_inv = cv2.bitwise_not(mask)

    # Extract background where cloak is present
    cloak_area = cv2.bitwise_and(background, background, mask=mask)

    # Extract current image where cloak is not present
    non_cloak_area = cv2.bitwise_and(image, image, mask=mask_inv)

    # Combine both
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    cv2.imshow("Red Cloak", final_output)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # pyright: ignore[reportAttributeAccessIssue]
    recorder= cv2.VideoWriter("Invisible.mp4",codec,20,(frame_width,frame_height))

    k = cv2.waitKey(5)
    if k == ord('q'):
        print("Quiting...")
        break

video.release()
cv2.destroyAllWindows()