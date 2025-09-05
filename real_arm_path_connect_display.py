import socket
import csv
import time

def send_script(script, host='192.168.1.30', port=30003):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.send(script.encode('utf-8'))
    s.close()

def load_points_from_csv(path):
    points = []
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            point = [float(x) for x in row if x.strip() != '']
            points.append(point)
    return points

def generate_script(points, with_initial=False):
    script = "def myProgram():\n"
    # if with_initial:
    #     point1 = [1.57,-1.57,1.57,0,1.57,0]
    #     script += f"  movej({point1}, a=1.4, v=0.5)\n"
    for i, point in enumerate(points):
        if i == 0 or i == len(points)-1:
            script += f"  movej({point}, a=1.4, v=0.1)\n"
        else:
            script += f"  movej({point}, a=1.4, v=0.1,r=0.001)\n"
    script += "end\n"
    return script

# 先執行第一段
# points1 = load_points_from_csv(r"C:\alan\webots\UR5e教學\joint_poses_4faceS_wp7cm_0708_final.csv")
points1 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_3_adjust4.0.csv")
# points1 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_3_adjust3.0.csv")
# points1 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_3_adjust3.0.csv")
# points1 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_final.csv")
script1 = generate_script(points1, with_initial=True)
send_script(script1)

# # 等機器執行完第一段（你可以估計大概的時間）
# time.sleep(77)  # 可依你機器的執行速度調整秒數

# # # # # 再執行第二段
# points2 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_2_adjust4.0.csv")
# script2 = generate_script(points2)
# send_script(script2)

# time.sleep(137)  # 可依你機器的執行速度調整秒數 

# # # # # 再執行第三段
# points3 = load_points_from_csv(r"C:\alan\webots\UR5e教學\adjust_path\0708_3_adjust4.0.csv")
# script3 = generate_script(points3)
# send_script(script3)