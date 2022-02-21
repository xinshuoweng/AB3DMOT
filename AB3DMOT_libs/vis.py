import numpy as np, cv2

def draw_box3d_image(image, qs, img_size=(900, 1600), color=(255,255,255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    def check_outside_image(x, y, height, width):
        if x < 0 or x >= width: return True
        if y < 0 or y >= height: return True

    # if 6 points of the box are outside the image, then do not draw
    pts_outside = 0
    for index in range(8):
        check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
        if check: pts_outside += 1
    if pts_outside >= 6: return image, False

    # actually draw
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0,4):
           i,j=k,(k+1)%4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

           i,j=k+4,(k+1)%4 + 4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

           i,j=k,k+4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return image, True
