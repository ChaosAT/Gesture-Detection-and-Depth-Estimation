import copy
import time
import ctypes

import cv2
import numpy as np

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import *


class ImgGetter(object):
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

        time.sleep(3)

    def getRGB(self, show=True, save=False):
        color_frame = self.kinect.get_last_color_frame()
        color_img = color_frame.reshape(
            (self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        color_img = color_img[:, :, :3]
        color_img = cv2.resize(color_img, (self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height))
        final_color_img = color_img[5:421, 42:458]
        if show:
            cv2.imshow("Normal RGB", color_img)
        if save:
            #保存RGB图
            pass
        return final_color_img

    def getDepth(self, show=True, saveNormal=False, saveRaw=False):
        depth_frame = self.kinect.get_last_depth_frame()
        depth_img = depth_frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width))
        final_depth_img = depth_img[5:421, 42:458]

        depth_img_view = copy.deepcopy(final_depth_img)
        depth_img_view = (depth_img_view / 8000 * 255).astype(np.uint8)

        if show:
            cv2.imshow('Test Depth View', depth_img_view)
        if saveNormal:
            #保存可视化深度图
            pass
        if saveRaw:
            #保存原始浮点深度图
            pass
        return final_depth_img, depth_img_view

    #获取通过深度图像对齐的RGB图像
    def getAlignRGB(self, show=True, save=False):
        align_color_img = self.color2depth(_ColorSpacePoint, self.kinect._depth_frame_data,
                                              return_aligned_image=True)
        final_align_color_img = align_color_img[5:421, 42:458]

        if show:
            cv2.imshow('Align Color Img', final_align_color_img)
        if save:
            #保存对齐的RGB图
            pass
        return final_align_color_img

    def color2depth(self, color_space_point, depth_frame_data, return_aligned_image=True):
        """
        :param kinect: kinect class
        :param color_space_point: _ColorSpacePoint from PyKinectV2
        :param depth_frame_data: kinect._depth_frame_data
        :param show: shows aligned image with color and depth
        :return: mapped depth to color frame
        """
        # Map Depth to Color Space
        depth2color_points_type = color_space_point * np.int(512 * 424)
        depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(color_space_point))
        self.kinect._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(512 * 424), depth_frame_data, self.kinect._depth_frame_data_capacity, depth2color_points)
        colorXYs = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(self.kinect.depth_frame_desc.Height * self.kinect.depth_frame_desc.Width,)))  # Convert ctype pointer to array
        colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
        colorXYs += 0.5
        colorXYs = colorXYs.reshape(self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width, 2).astype(np.int)
        colorXs = np.clip(colorXYs[:, :, 0], 0, self.kinect.color_frame_desc.Width - 1)
        colorYs = np.clip(colorXYs[:, :, 1], 0, self.kinect.color_frame_desc.Height - 1)
        color_frame = self.kinect.get_last_color_frame()
        color_img = color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)

        align_color_img = np.zeros((424, 512, 4), dtype=np.uint8)
        align_color_img[:, :] = color_img[colorYs, colorXs, :]
        align_color_img = align_color_img[:, :, :3]

        if return_aligned_image:
            return align_color_img
        return colorXs, colorYs

if __name__ == '__main__':

    kinect_img_getter = ImgGetter()
    rgb = kinect_img_getter.getRGB()
    depth = kinect_img_getter.getDepth()
    align_rgb = kinect_img_getter.getAlignRGB()

    cv2.waitKey(0)
    cv2.destroyAllWindows()