
# 导入相关库
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# 窗口初始化
def init():
    # 观察坐标系原点
    x0, y0, z0 = 5.0, 5.0, 7.0
    # 观察坐标系视点
    xviw, yviw, zviw = 0.0, 0.0, 0.0
    # 观察坐标系向量
    Vx, Vy, Vz = 0.0, 1.0, 0.0
    # 裁剪窗口坐标范围
    xMin, yMin, xMax, yMax = -1.0, -1.0, 1.0, 1.0

    # 前后裁剪面深度
    dfont, dback = 2.0, 30.0
    glClearColor(1.0, 1.0, 1.0, 0.0)
    # 三维观察参数
    gluLookAt(x0, y0, z0, xviw, yviw, zviw, Vx, Vy, Vz)
    glMatrixMode(GL_MODELVIEW)
    # 放缩变换
    #glScalef(2.0, 2.0, 3.0)
    glScalef(1, 1, 1)
    # 旋转变换
    #glRotatef(45.0, 0.0, 1.0, 1.0)
    glRotatef(0.0, 0.0, 1.0, 1.0)
    # 投影变换
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # 透视投影
    glFrustum(xMin, xMax, yMin, yMax, dfont, dback)

def draw_init(display):
    display = display
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0, 8, 3, 0, 180, 0, 0, 0, 1)
    viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    glLoadIdentity()
    return viewMatrix

def drawCoordinate():
    '''
    绘制三维的坐标系，并绘制由坐标轴构成的平面的网格，各个坐标轴的颜色以及由坐标轴所引出的网格线的颜色为：
    x: (1.0, 0.0, 0.0)
    y: (0.0, 1.0, 0.0)
    z: (0.0, 0.0, 1.0)
    :return:
    '''
    #设置网格线间的步长
    step = 0.2
    #设置网格线的数量、长度
    line_num = 15
    line_len = 4
    grid_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    glLineWidth(3)
    origin = [0.0, 0.0, 0.0]
    #画出xyz轴
    for i in range(3):
        tmp = [0.0, 0.0, 0.0]
        tmp[i] = line_len*1.02
        glColor3f(*grid_color[i])
        glBegin(GL_LINES)
        glVertex3f(*origin)
        glVertex3f(*tmp)
        glEnd()


    # 画出由坐标轴构成的平面的网格
    glLineWidth(1)
    for i in range(3):
    #每次把一个坐标轴上引出的网格线都画出来，所以起点都是一致的
        start = [0.0, 0.0, 0.0]
        glColor3f(*grid_color[i])
        for j in range(line_num):
            end = [0.0, 0.0, 0.0]
            glBegin(GL_LINES)
            start[i] = start[i] + step
            end[i] = start[i]
            for k in {0,1,2} - {i,}:
                end[k] = line_len
                glVertex3f(*start)
                glVertex3f(*end)
                end[k] = 0.0
            glEnd()
def drawPlane():
    glColor4f(0.5, 0.5, 0.5, 1)
    glBegin(GL_QUADS)
    glVertex3f(-10, -10, -2)
    glVertex3f(10, -10, -2)
    glVertex3f(10, 10, -2)
    glVertex3f(-10, 10, -2)
    glEnd()

def draw_points():
    # 绿色
    glColor3f(0.0, 1.0, 0.0)
    # 绘制立方体
    glPointSize(5.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(3, 2, 3)
    glEnd()

    glBegin(GL_LINES)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.3, 0.5, 0.3)
    glEnd()
    # glutSolidSphere(1.0,)
    # 前景色为黑色
    glColor3f(0.0, 0.0, 0.0)
    # 线宽
    glLineWidth(1.0)
    # 线框
    # glutWireCube(1.0)
# 回调函数
def display():
    glClear(GL_COLOR_BUFFER_BIT)
    drawCoordinate()
    draw_points()
    #glEnd()
    glFlush()


if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    # 设置窗口大小
    glutInitWindowSize(800, 800)
    glutCreateWindow("The Perspective Projection")
    init()
    # 显示回调函数
    glutDisplayFunc(display)
    # 开始循环 执行程序
    glutMainLoop()

