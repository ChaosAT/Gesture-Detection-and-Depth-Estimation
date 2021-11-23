import os
import time

import cv2
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import kinect.ImgGetter as Kinect_Img_Getter
import yolov3_depth.Detector as Yolo_Depth_Detector
import draw.drawpoints as drawpoints

SHOW_DEPTH_VIEW = False

def kinect_init():
    kinect_img_getter = Kinect_Img_Getter.ImgGetter()
    return kinect_img_getter
def yolo_init(model_type):
    opt = Yolo_Depth_Detector.args_init(model_type).parse_args()
    yolo_depth_detector = Yolo_Depth_Detector.Detector(opt=opt)
    return yolo_depth_detector

def draw_points(kinect_img_getter, yolo_depth_detector):

    global point_list

    # 绿色
    #current_point = detect(yolo_model, fd_model, kinect)
    glColor3f(1.0, 1.0, 1.0)
    glPointSize(10.0)

    point_num = len(point_list)

    #使用检测算法获取点,没有目标时为none
    #next_point = (np.random.rand() * 3, np.random.rand() * 3, np.random.rand() * 3)

    # 获取kinect的目标真实深度
    true_depth_img, depth_img_view = kinect_img_getter.getDepth()
    if SHOW_DEPTH_VIEW:
        depth_img_view = np.stack((depth_img_view, depth_img_view, depth_img_view), axis=2)
    next_point = None
    target_class_idx = -1
    align_rgb = kinect_img_getter.getAlignRGB(show=False)
    result, result_img = yolo_depth_detector.detect(align_rgb, showResultImg=True)
    if len(result) != 0:
        target_class_idx, target_conf, target_depth, target_xmid, target_ymid = yolo_depth_detector.parse_result(result)

        if SHOW_DEPTH_VIEW:
            #cv2.circle(depth_img_view, (target_xmid, target_ymid), 2, color=(255, 255, 255))
            #depth_img_view[target_ymid][target_xmid] = [255,255,255]
            cv2.imshow("Tt", depth_img_view)
        true_depth = true_depth_img[target_ymid][target_xmid]
        #print(target_depth, true_depth)
        #target_depth = true_depth
        target_xmid = 4 - target_xmid / 125
        target_ymid = 4 - target_ymid / 125

        target_depth = (target_depth-700)/500*3
        #target_depth = 0
        next_point = np.array([target_xmid, target_depth, target_ymid])



    # 使用FastDepth
    #next_point = np.array([5 - target_xmid / 121.6, grey_depth_pred[int(target_xmid / 2.72)][int(target_ymid / 2.72)] / 51, 5 - target_ymid / 121.6])

    if target_class_idx == 3:
        #end
        if point_num == 0:
            # 第一个点
            point_list.append(next_point)
            print("Start.")
    elif target_class_idx == 0 or target_class_idx == 1 or target_class_idx == 2:

        if type(next_point) == np.ndarray:
            if point_num == 0:
                #第一个点，需用start手势触发
                pass
            else:
                point_distance = np.sqrt(np.sum(np.square(next_point-point_list[-1])))
                point_depth_distance = np.abs(next_point[1]-point_list[-1][1])
                point_xy_distance = np.square(next_point[0]-point_list[-1][0])+np.square(next_point[2]-point_list[-1][2])
                point_xy_distance = np.sqrt(point_xy_distance)
                print("All distance:",point_distance, "XY distance:",point_xy_distance, "Depth distance:",point_depth_distance)

                if point_distance > 0.5:
                    #print("重复点")
                    pass
                if point_distance < 0.1:
                    #print("重复点")
                    pass
                else:
                    #可用点
                    if point_depth_distance < 0.1:
                        next_point[1]=point_list[-1][1]
                        pass
                    if point_depth_distance > 0.4:

                        pass
                    else:
                        if next_point[1]-point_list[-1][1]>0:
                            next_point[1] = point_list[-1][1]+0.05
                        else:
                            next_point[1] = point_list[-1][1]-0.05
                    #else:
                    point_list.append(next_point)
    else:
        #print("None.")
        pass

    #绘制点列表中的所有点
    glBegin(GL_LINE_STRIP)
    for j in range(point_num):
        if j+1 == point_num:
            break
        glVertex3f(point_list[j][0], point_list[j][1], point_list[j][2])
        glVertex3f(point_list[j+1][0], point_list[j+1][1], point_list[j+1][2])
    glEnd()

def test(show = True):
    TEST_RGB_DIR = "E:/bishe2/YOLOv3-complete-pruning-master (2)/data/images/test"
    test_img_list = os.listdir(TEST_RGB_DIR)
    for i in range(len(test_img_list)):
        test_img_list[i] = os.path.join(TEST_RGB_DIR, test_img_list[i])

    for test_img_name in test_img_list:
        img = cv2.imread(test_img_name)
        label = yolo_depth_detector.detect(img, showResultImg=True)
        if show:
            cv2.waitKey(0)
def FPS_test():
    kinect_img_getter = kinect_init()
    yolo_depth_detector = yolo_init("tinyyolo")

    test_rgb = cv2.imread("1.jpg")
    FPSs = []
    i = 0
    while i != 2000:
        start_time = time.time()

        result, result_img = yolo_depth_detector.detect(test_rgb, showResultImg=False)
        target_class, target_conf, target_depth, target_xmid, target_ymid = yolo_depth_detector.parse_result(result)
        print(target_class, target_conf, target_depth, target_xmid, target_ymid)

        # rgb = kinect_img_getter.getRGB(show=True)
        # result, result_img = yolo_depth_detector.detect(rgb, showResultImg=False)

        # align_rgb = kinect_img_getter.getAlignRGB(show=False)
        # result, result_img = yolo_depth_detector.detect(align_rgb, showResultImg=False)

        end_time = time.time()
        FPS = int(1 / (end_time - start_time))
        FPSs.append(FPS)
        result_img = np.array(result_img)
        cv2.putText(result_img, "FPS:{0}".format(FPS), (5, 25), 0, 1, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("Result", result_img)
        cv2.waitKey(1)
        i += 1
    FPSs = np.array(FPSs)
    print(np.mean(FPSs))

def showdata():
    ''''''
    im1 = cv2.imread("example_rgb/1.jpg")
    #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread("example_rgb/2.jpg")
    #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3 = cv2.imread("example_rgb/3.jpg")
    #im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im4 = cv2.imread("example_rgb/4.jpg")
    #im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)

    _, im1 = yolo_depth_detector.detect(im1, showResultImg=True)
    cv2.waitKey(0)
    _, im2 = yolo_depth_detector.detect(im2, showResultImg=True)
    _, im3 = yolo_depth_detector.detect(im3, showResultImg=True)
    _, im4 = yolo_depth_detector.detect(im4, showResultImg=True)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    '''
    im1 = np.loadtxt("example_depth/1.txt")
    im1 = (im1 / 8000 * 255.0).astype(np.uint8)
    im2 = np.loadtxt("example_depth/2.txt")
    im2 = (im2 / 8000 * 255.0).astype(np.uint8)
    im3 = np.loadtxt("example_depth/3.txt")
    im3 = (im3 / 8000 * 255.0).astype(np.uint8)
    im4 = np.loadtxt("example_depth/4.txt")
    im4 = (im4 / 8000 * 255.0).astype(np.uint8)
    '''

    plt.figure()
    plt.suptitle("Gesture Depth Data")
    plt.subplot(1, 4, 1), plt.title("Draw")
    plt.imshow(im1), plt.axis('off')
    plt.subplot(1, 4, 2), plt.title("Start")
    plt.imshow(im2), plt.axis('off')
    plt.subplot(1, 4, 3), plt.title("Erase")
    plt.imshow(im3), plt.axis('off')
    plt.subplot(1, 4, 4), plt.title("Stop")
    plt.imshow(im4), plt.axis('off')
    plt.show()

if __name__ == '__main__':

    #kinect_img_getter = kinect_init()
    yolo_depth_detector = yolo_init("tinyyolo")
    showdata()
    #test()

    '''
    global point_list
    point_list = []

    pygame.init()
    display = (400, 300)
    scree = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    viewMatrix = drawpoints.draw_init(display)

    # init mouse movement and center mouse on screen
    displayCenter = [scree.get_size()[i] // 2 for i in range(2)]
    mouseMove = [0, 0]
    pygame.mouse.set_pos(displayCenter)

    up_down_angle = 0.0
    paused = False
    run = True

    FPSs = []
    while run:

        start_time = time.time()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    run = False
                if event.key == pygame.K_PAUSE or event.key == pygame.K_p:
                    paused = not paused
                    pygame.mouse.set_pos(displayCenter)
            if not paused:
                if event.type == pygame.MOUSEMOTION:
                    mouseMove = [event.pos[i] - displayCenter[i] for i in range(2)]
                pygame.mouse.set_pos(displayCenter)

        if not paused:
            # get keys

            keypress = pygame.key.get_pressed()
            # mouseMove = pygame.mouse.get_rel()

            # init model view matrix
            glLoadIdentity()

            # apply the look up and down
            up_down_angle += mouseMove[1] * 0.1
            glRotatef(up_down_angle, 1.0, 0.0, 0.0)

            # init the view matrix
            glPushMatrix()
            glLoadIdentity()

            # apply the movment
            if keypress[pygame.K_w]:
                glTranslatef(0, 0, 0.15)
            if keypress[pygame.K_s]:
                glTranslatef(0, 0, -0.15)
            if keypress[pygame.K_d]:
                glTranslatef(-0.15, 0, 0)
            if keypress[pygame.K_a]:
                glTranslatef(0.15, 0, 0)

            # apply the left and right rotation
            glRotatef(mouseMove[0] * 0.5, 0.0, 1.0, 0.0)

            # multiply the current matrix by the get the new view matrix and store the final vie matrix
            glMultMatrixf(viewMatrix)
            viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

            # apply view matrix
            glPopMatrix()
            glMultMatrixf(viewMatrix)

            glLightfv(GL_LIGHT0, GL_POSITION, [1, -1, 1, 0])

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glPushMatrix()

            drawpoints.drawPlane()
            drawpoints.drawCoordinate()
            draw_points(kinect_img_getter, yolo_depth_detector)

            glPopMatrix()

            pygame.display.flip()

            end_time = time.time()
            #FPS = int(1 / (end_time - start_time))
            #FPSs.append(FPS)

            pygame.time.wait(1)

    pygame.quit()
    #FPSs = np.array(FPSs)
    #print(np.mean(FPSs))
    '''
