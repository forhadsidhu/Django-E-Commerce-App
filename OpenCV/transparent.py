from PIL import Image
import  cv2 as cv
img = Image.open('download.jpg')
img = img.convert("RGBA")

pixdata = img.load()

width, height = img.size
for y in range(height):
    for x in range(width):
        if pixdata[x, y] == (255, 255, 255, 255):
            pixdata[x, y] = (255, 255, 255, 0)

cv.imshow('test_image',img)
cv.waitKey(1000)
img.save("img2.png", "PNG")
