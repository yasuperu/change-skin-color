import cv2 as cv
import numpy as np
import random

# Capture webcam footage
cap = cv.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    cv.imwrite('frames' + '.png', frame)
    key = cv.waitKey(20)
    if key == 27:  # exit on ESC
        break

    # Load the aerial image and convert to HSV colourspace
    image = cv.imread("frames.png")
    # Load image for second time to add transparency
    image_rgba = cv.imread("frames.png")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "caucasian skin"
    skin_lo = np.array([0, 48, 80])
    skin_hi = np.array([20, 255, 255])

    # Mask image to only select skin
    mask = cv.inRange(hsv, skin_lo, skin_hi)


    # Change image color where we found skin
    mask_alpha = np.zeros(image.shape, dtype=image.dtype)
    #mask_alpha[mask > 0] = (random.choice(skin_colors))
    mask_alpha[mask > 0] = (1, 4, 10)

    # Make copy of image in seperate channels of B, G and R
    b_chan, g_chan, r_chan = cv.split(mask_alpha)
    # My temporary grayscale of image to mask out the black
    tmp = cv.cvtColor(mask_alpha, cv.COLOR_BGR2GRAY)
    # Define alpha layer
    a_chan = np.full(b_chan.shape, 1, dtype=b_chan.dtype) * 50
    # Delete black in layer
    _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)
    new_alpha = cv.multiply(a_chan, alpha)
    # Adjust the alpha brightness
    new_alpha2 = cv.multiply(0.6, new_alpha)
    # Merge Layers
    mask_alpha_half = cv.merge((b_chan, g_chan, r_chan, new_alpha2))


    def overlay_transparent(background, overlay, x, y):

        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
                ],
                axis=2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

        return background


    # Create image from both
    overlay_transparent(image_rgba, mask_alpha_half, 0, 0)

    cv.imwrite("result.png", image_rgba)
    img = cv.imread('result.png')
    cv.imshow('result',img)

cap.release()
cv.destroyAllWindows()