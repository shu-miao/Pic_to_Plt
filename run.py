import cv2
import numpy as np
import matplotlib.pyplot as plt


def pic_plt(pic_file, plt_file, color_l, color_h, size):
    '''
    把图片中的特定颜色重新绘制为matplotlib的散点图
    :param pic_file: 传入图片路径
    :param plt_file: 保存名称
    :param color_l: 色值min值
    :param color_h: 色值max值
    :param size: 点大小
    :return: 保存路径
    '''
    img = cv2.imread(pic_file)
    img = cv2.flip(img, flipCode=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_list_l = np.array([color_l])
    color_list_h = np.array([color_h])
    mask = cv2.inRange(hsv, color_list_l, color_list_h)
    # res = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    xy = np.column_stack(np.where(mask))
    x, y = [], []
    for i in xy:
        y.append(i[0])
        x.append(i[1])
    plt.scatter(x, y, s=size)
    plt.savefig(f'{plt_file}.jpg')
    print(f'图片保存到{plt_file}.jpg')
    plt.show()
    return plt_file + '.jpg'


pic_plt(pic_file='img.jpg', plt_file='out', color_l=[156, 43, 46], color_h=[180, 255, 255], size=0.5)
