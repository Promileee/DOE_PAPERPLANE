import pyautogui
import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import pandas as pd
import math

########################################################################################################################

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\tesseract\tesseract.exe'
"""
这里是tessetact图像识别库的调用，具体可以询问AI模型
"""

START_POSITION={"x":1304, "y":1153}
"""
这个常量是游戏初始界面开始游戏的坐标
"""

DETECTION_REGION = {
    'left': 410,
    'top': 686,
    'width': 978 - 410,
    'height': 1037 - 686
}
"""
这个常量是参数设置界面weight与elevator两个参数调整框在屏幕上的像素范围
注意这个范围需要根据你的电脑的情况而定
具体范围要求并不严格，但是一定要仅包含两个橙色的块，不能有其他橙色元素的出现
"""

PLANE_A_POSITION = {"x":494, "y":598}
PLANE_B_POSITION = {"x":666, "y":598}
PLANE_C_POSITION = {"x":847, "y":598}
"""
这个常量是参数设置界面三个飞机类型对应按钮的坐标
"""

WINGLETS_ON_POSITION = {'x':825,'y':1158}
WINGLETS_OFF_POSITION = {'x':552,'y':1158}
"""
这个常量是参数设置界面WINGLETS开和关对应的坐标
"""

PRACTICE_POSITION={'x':1841,'y':936}
"""
这个常量是参数设置界面PRACTICE按钮的坐标
"""

INITIAL_POSITION={'x':829,'y':879}
"""
这个常量是游戏操作界面中飞机初始位置的坐标，就是你要去飞飞机，首先要到这个点，再拖动
"""

PLAY_AGAIN_POSITION={'x':1249,'y':724}
"""
这个常量是飞行完成后，PLAY AGAIN按钮的坐标
"""

SCORE_DETECTION_REGION = {
        'left': 1980,
        'top': 1150,
        'width': 2104 - 1980,
        'height': 1300 - 1225
    }
"""
这个常量是游戏操作界面游戏得分的坐标范围，越精准越好，得分识别对这个范围很敏感，可以截一张全屏截图，然后再Photoshop中确定坐标范围
"""

SET_PARA_POSITION = {'x':430,'y':1290}
"""
这个常量是一次飞行完成后再次设置参数的按钮的坐标（就是那个小齿轮）
"""



########################################################################################################################

def capture_screen():
    region = (
        DETECTION_REGION['left'],
        DETECTION_REGION['top'],
        DETECTION_REGION['width'],
        DETECTION_REGION['height']
    )
    return cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_RGB2BGR)
"""
这个函数用于截取weight与elevator参数调整区域的截图
"""

def find_orange_blocks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if 200 < cv2.contourArea(cnt) < 10000]
    contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))  # 按Y/X排序

    blocks = []
    for cnt in contours[:2]:
        x, y, w, h = cv2.boundingRect(cnt)
        blocks.append((x, y, w, h))
    return blocks
"""
这个函数用于找到当前状态下两个橙色块的位置
"""

def ocr_number(full_screen_image, block):
    x, y, w, h = block
    global_x = x + DETECTION_REGION['left']
    global_y = y + DETECTION_REGION['top']

    left = global_x - 5
    upper = max(0, global_y - h - 12)
    right = global_x + w
    lower = global_y

    number_region = full_screen_image.crop((left, upper, right, lower))
    number_region = number_region.convert('L')
    img_array = np.array(number_region)
    _, binary = cv2.threshold(img_array, 90, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary_pil = Image.fromarray(binary)

    # 保存调试图像（用于OCR调试）
    #binary_pil = Image.fromarray(binary)
    #binary_pil.save(f"debug_ocr_{int(time.time())}.png")
    """
    前期调试时建议取消上面两行注释，调试完成后可以注释
    """

    text = pytesseract.image_to_string(
        binary_pil,
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    )
    return int(text.strip()) if text.strip().isdigit() else 0
"""
这个函数用于识别weight与elevator当前值
"""

def drag_and_adjust(target_value, initial_block_coords):
    x, y, w, h = initial_block_coords
    global_x = x + DETECTION_REGION['left'] + w // 2
    global_y = y + DETECTION_REGION['top'] + h // 2

    pyautogui.moveTo(global_x, global_y, duration=0.1)
    #pyautogui.moveTo(global_x, global_y)
    pyautogui.mouseDown(button='left')

    step = 1
    direction = 1

    while True:
        #pyautogui.moveRel(direction * step, 0, duration=0.1)
        pyautogui.moveRel(direction * step, 0,duration=0.1)
        #time.sleep(0.1)

        current_screen_area = capture_screen()
        current_blocks = find_orange_blocks(current_screen_area)
        current_block = find_closest_block(initial_block_coords, current_blocks)

        if not current_block:
            print("未检测到方块，停止拖拽")
            break

        current_number = ocr_number(pyautogui.screenshot(), current_block)
        print(f"当前数值：{current_number}，目标：{target_value}")
        if -10 < current_number - target_value < 10:
            step = 1
        else:
            step = 18

        if current_number == target_value:
            break
        elif (current_number > target_value) and (direction == 1) or \
                (current_number < target_value and direction == -1):
            direction *= -1
        else:
            pass

    pyautogui.mouseUp(button='left')
"""
这个函数用于拖动调整weight与elevator至目标值
"""

def find_closest_block(initial_block, blocks):
    if not blocks:
        return None
    initial_x, initial_y, _, _ = initial_block
    min_distance = float('inf')
    closest_block = None
    for block in blocks:
        x, y, w, h = block
        distance = ((x - initial_x) ** 2 + (y - initial_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_block = block
    return closest_block

def get_block_number(block):
    x, y, w, h = block
    global_x = DETECTION_REGION['left'] + x + w // 2
    global_y = DETECTION_REGION['top'] + y + h // 2

    #pyautogui.moveTo(global_x, global_y, duration=0.1)
    pyautogui.moveTo(global_x, global_y)
    pyautogui.mouseDown(button='left')
    current_screen = pyautogui.screenshot()
    number = ocr_number(current_screen, block)
    pyautogui.mouseUp(button='left')
    return number

def weight_and_elevator(target_values):
    #target_values = [50, 75]  # 目标数值

    while True:
        detection_area = capture_screen()
        blocks = find_orange_blocks(detection_area)

        if len(blocks) != 2:
            print("未检测到两个方块，重新检测...")
            continue

        block1, block2 = blocks  # 按Y排序后的结果

        current_number1 = get_block_number(block1)
        current_number2 = get_block_number(block2)
        print(f"当前数值：Block1={current_number1}, Block2={current_number2}")



        if current_number1 == target_values[0] and current_number2 == target_values[1]:
            print("已达到目标值，完成！")
            break

        if current_number1-target_values[0]<-50:
            drag_and_adjust(target_values[0]-50, block1)
        elif current_number1-target_values[0]>50:
            drag_and_adjust(target_values[0]+50, block1)
        else:
            drag_and_adjust(target_values[0], block1)
        if current_number2-target_values[1]<-50:
            drag_and_adjust(target_values[1]-50, block2)
        elif current_number2-target_values[1]>50:
            drag_and_adjust(target_values[1]+50,block2)
        else:
            drag_and_adjust(target_values[1], block2)

def game_start():
    pyautogui.click(START_POSITION['x'], START_POSITION['y'])
    time.sleep(1.2)
    """
    游戏启动有过度动画，这里使用延时
    """

def plane_type(target_plane):
    if target_plane == 'A':
        pyautogui.moveTo(PLANE_A_POSITION['x'],PLANE_A_POSITION['y'])
        pyautogui.click(PLANE_A_POSITION['x'],PLANE_A_POSITION['y'])
    elif target_plane == 'B':
        pyautogui.moveTo(PLANE_B_POSITION['x'],PLANE_B_POSITION['y'])
        pyautogui.click(PLANE_B_POSITION['x'],PLANE_B_POSITION['y'])
    else:
        pyautogui.moveTo(PLANE_B_POSITION['x'],PLANE_B_POSITION['y'])
        pyautogui.click(PLANE_B_POSITION['x'],PLANE_B_POSITION['y'])

def winglets(target_wins):
    if target_wins == 'ON':
        pyautogui.moveTo(WINGLETS_OFF_POSITION['x'],WINGLETS_OFF_POSITION['y'])
        pyautogui.dragTo(WINGLETS_ON_POSITION['x'],WINGLETS_ON_POSITION['y'],duration=0.2)
    else:
        pyautogui.moveTo(WINGLETS_ON_POSITION['x'],WINGLETS_ON_POSITION['y'])
        pyautogui.dragTo(WINGLETS_OFF_POSITION['x'],WINGLETS_OFF_POSITION['y'])

def para(target_para):
    target_plane = target_para[0]
    target_value = target_para[1:3]
    target_wins = target_para[3]
    plane_type(target_plane)
    weight_and_elevator(target_value)
    winglets(target_wins)

def practice():
    pyautogui.click(PRACTICE_POSITION['x'], PRACTICE_POSITION['y'])

def throw(target_position):
    x = target_position[0]
    y = target_position[1]
    pyautogui.moveTo(INITIAL_POSITION['x'], INITIAL_POSITION['y'])
    pyautogui.mouseDown()
    pyautogui.moveTo(x,y,duration=0.3)
    time.sleep(0.3)
    pyautogui.mouseUp()

def play_again():
    pyautogui.click(1249,724)
    time.sleep(1.2)

def score():
    time.sleep(13)
    """
    在指定范围内采集白色数字并识别。

    返回:
        int: 识别到的数字。如果无法识别，则返回 None。
    """
    # 定义检测区域
    detection_region = SCORE_DETECTION_REGION

    # 截取屏幕指定区域
    region = (
        detection_region['left'],
        detection_region['top'],
        detection_region['width'],
        detection_region['height']
    )
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 转换为灰度图像
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # 创建白色掩码（RGB 值为 255, 255, 255）
    lower_white = np.array([254, 254, 254])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(screenshot, lower_white, upper_white)

    # 应用掩码提取白色区域
    white_only = cv2.bitwise_and(gray, gray, mask=mask)

    # 对白色区域进行二值化处理
    _, binary = cv2.threshold(white_only, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作：去噪和增强数字边界
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    # 将处理后的图像转换为 PIL 图像
    binary_pil = Image.fromarray(binary)

    # 保存调试图像（用于OCR调试）
    binary_pil.save(f"debug_ocr_{int(time.time())}.png")

    # 使用 Tesseract 进行 OCR 识别
    try:
        text = pytesseract.image_to_string(
            binary_pil,
            config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if text.isdigit():  # 确保识别结果是数字
            #print(int(text))
            return float(int(text)) * 0.1
        else:
            print("OCR 识别结果非数字:", text)
            return None
    except Exception as e:
        print("OCR 识别失败:", e)
        return None

def set_para():
    pyautogui.moveTo(SET_PARA_POSITION['x'], SET_PARA_POSITION['y'])
    pyautogui.click(SET_PARA_POSITION['x'], SET_PARA_POSITION['y'])


def experiment(target_para, target_position):
    """
    在打开游戏参数设置界面后执行投掷操作并返回得分
    :param target_para:
    :param target_position:
    :return:
    """
    #target_para = ['B', 50, 37, 'ON']
    para(target_para)
    practice()
    #target_position = [610,970]
    throw(target_position)
    ex_score = score()
    return ex_score

def cal_position(target):
    angle = target[0]
    force = target[1]
    angle_radians = math.radians(angle)
    sin_value = math.sin(angle_radians)
    cos_value = math.cos(angle_radians)
    delta_x = force * cos_value
    delta_y = force * sin_value
    x = INITIAL_POSITION['x'] - round(delta_x) + np.random.normal(loc=0,scale=25,size=1)
    y = INITIAL_POSITION['y'] + round(delta_y) + np.random.normal(loc=0,scale=25,size=1)
    """
    这里加入了一个正态随机数，用于模拟随机因素对试验带来的影响，可删除
    如果随机因素太大，可以调整Scale参数，这个参数就是正态分布的方差
    同样，随机因素太小同理
    """
    return [x,y]

def main():
    time.sleep(3)
    first_start = False
    #paper_plane_type = ['A', 'B', 'C']
    paper_plane_type = ['A']
    #weight = [0, 20, 40, 60, 80, 100]
    weight = [50]
    elevator = [0, 20, 40, 60, 80, 100]
    #elevator = [50]
    #plane_winglets = ['ON', 'OFF']
    plane_winglets = ['ON']
    '''
    target_position = [
        [732,940]
    ]
    '''
    #angle = [15, 22.5, 30, 37.5, 45]
    angle = [15, 17.5, 20, 22.5,25, 27.5, 30, 32.5, 35, 37.5, 40]
    #force = [100,120,140,160,180,200]
    force = np.linspace(200,400,20)
    #force = [40, 120]

    data = []
    for i in range(len(paper_plane_type)):
        for j in range(len(weight)):
            for k in range(len(elevator)):
                for l in range(len(plane_winglets)):
                    if first_start:
                        game_start()
                        first_start = False
                    else:
                        set_para()

                    time.sleep(0.5)

                    target_para = [
                        paper_plane_type[i],
                        weight[j],
                        elevator[k],
                        plane_winglets[l],
                    ]

                    para(target_para)
                    practice()

                    for o in range(len(angle)):
                        for p in range(len(force)):
                            target_throw = [angle[o], force[p]]
                            target_position = cal_position(target_throw)
                            throw(target_position)
                            ex_score = score()
                            if ex_score < 10:
                                play_again()
                                break
                            data_temp = target_para + target_throw + [ex_score]
                            data.append(data_temp)
                            play_again()

    # 转换为 DataFrame，并指定列名
    df = pd.DataFrame(data, columns=["type", "weight", "elevator", "winglets", "angle", "force","score"])

    # 导出 Excel，保留表头
    df.to_excel(f"output{int(time.time())}.xlsx", index=False)



if __name__ == '__main__':
    main()