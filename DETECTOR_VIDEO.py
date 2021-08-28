import cv2
import matplotlib.pylab as plt
import numpy as np
# lets mask the region other than roi
def region_of_interest(img,vertices):
    mask=np.zeros_like(img)
    # channel_count=img.shape[2] because we had coverted our image in grayscale so it has only one channel count
    # match_mask_color=(255,)*channel_count
    match_mask_color=255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image
def draw_the_lines(img,lines):
    img-np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),8)
    img=cv2.addWeighted(img,0.8,blank_image,1,0.0)
    return img
def process(image):
    print(image.shape)
    height=image.shape[0]
    width=image.shape[1]
    # we require bottom left top center and bottom right
    # region_of_interest_vertices=[(0,height),(width/2,height/2),(width-1,height)]
    region_of_interest_vertices=[(0,height),(732,371),(width-250,height)]
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny_image=cv2.Canny(gray_image,100,200,)
    cropped_image=region_of_interest(canny_image,np.array([region_of_interest_vertices],np.int32))
    lines=cv2.HoughLinesP(cropped_image,6,np.pi/50,160,np.array([]),minLineLength=40,maxLineGap=25)
    image_with_lines=draw_the_lines(image,lines)
    return image_with_lines

cap=cv2.VideoCapture('123.mp4')
while(cap.isOpened()):
    ret,frame=cap.read()
    frame=process(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


