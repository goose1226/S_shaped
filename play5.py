# 鏡頭 Live 到網頁，
# 網頁控制鏡頭右轉、左轉
# 網頁控制鏡頭前進加速、前進減速、前進右轉、前進左轉
# 網頁增加「沿線自走」

import cv2
import time
from LOBOROBOT2 import LOBOROBOT  # 載入驅動智慧小車的基本運動庫函數
import numpy as np

import threading
import libcamera
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify

# ------------------------------------------------------------------
INDEX_HTML = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>MJPEG from Directory</title>
  <style>
    img { border-radius:8px; border:1px solid #333; }
  </style>
</head>
<body align=center>
  <h3>Motion JPEG from Pi 小車影像</p>
  <img src="/live" alt="MJPEG Stream">&nbsp;&nbsp;&nbsp;
  <img src="/live2" alt="MJPEG Stream">
  <hr width='70%'>
  <button onclick="fetch('/api/turn_left', {method:'POST'})">鏡頭左轉</button>
  <button onclick="fetch('/api/turn_right',{method:'POST'})">鏡頭右轉</button>
  <button onclick="fetch('/api/turn_up',  {method:'POST'})">鏡頭上轉</button>
  <button onclick="fetch('/api/turn_down',{method:'POST'})">鏡頭下轉</button>
  <hr width='70%'>
  <button id="start"  onclick="fetch('/api/start',{method:'POST'})">開始</button>
  <button id="stop"   onclick="fetch('/api/stop', {method:'POST'})">停止</button>
  <button id="accel"  onclick="fetch('/api/accelerate', {method:'POST'})">前進加速</button>
  <button id="decel"  onclick="fetch('/api/decelerate', {method:'POST'})">前進減速</button>
  <button id="leftf"  onclick="fetch('/api/left_forward', {method:'POST'})">前進左轉</button>
  <button id="rightf" onclick="fetch('/api/right_forward',{method:'POST'})">前進右轉</button><br>
  <button id="autogo" onclick="fetch('/api/autogo',{method:'POST'})">沿線自走</button>
  <script>
    // 鍵盤快捷：W/S 控速，A/D 轉向，Space 停止
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (['w','W','ArrowUp'].includes(e.key)) document.getElementById('accel').click();
      if (['s','S','ArrowDown'].includes(e.key)) document.getElementById('decel').click();
      if (['a','A','ArrowLeft'].includes(e.key)) document.getElementById('leftf').click();
      if (['d','D','ArrowRight'].includes(e.key)) document.getElementById('rightf').click();
      if (['g','G'].includes(e.key)) document.getElementById('autogo').click();
      if (e.key === ' ') document.getElementById('stop').click();
    });
  </script>
</body>
</html>
"""

ww, hh, qq, angle, updown, Cam_X, Cam_Y = 320, 240, 50, 90, 20, 10, 9
slope = hh / ww
MIN_ANGLE = 20     # 為保守起見，別打到機械限位；你可改成 0
MAX_ANGLE = 150    # 同上；你可改成 180
MIN_UPDOWN = -2.5  # 為保守起見，別打到機械限位；你可改成 0
MAX_UPDOWN = 40    # 同上；你可改成 50
STEP_ANGLE = 5

speed, l_ofs, r_ofs, car_go, auto_go, cnt, lost = 20, 0, 0, 0, 0, 0, 0
SPEED_MIN   = 0
SPEED_MAX   = 100  # 依你的 PWM/速度設計調整
SPEED_STEP  = 5    # 每次加/減速度幅度

# ------------------------------------------------------------------
# ---------------------- Picamera2 初始化 ---------------------------
picamera = Picamera2()
config = picamera.create_preview_configuration(
            main={"format": "RGB888", "size": (ww, hh)},
            raw={"format": "SRGGB12", "size": (ww, hh)},
)
config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picamera.configure(config)
picamera.start()

clbrobot = LOBOROBOT()
clbrobot.t_stop(0.1)

# ------------------------------------------------------------------
# -- 鏡頭部分，使用執行緒(threading) 一種在單一程式內同時執行多個任務的技術 --
running = True
latest_frame = None
frame_lock = threading.Lock()       # 準備一個 Lock
angle_lock = threading.Lock()       
speed_lock = threading.Lock()

# 定義一個副程式，找出圖片中所有「線」
def get_canny(img):
    #global cnt
    output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(f'a{cnt}.png', img)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = cv2.erode(output, kernel)      # 侵蝕，將白色小圓點移除
    output = cv2.dilate(output, kernel)     # 膨脹
    output = cv2.medianBlur(output, 15)     # 模糊化，去除雜訊
    #cv2.imwrite(f'b{cnt}.png', output)
    
    output = cv2.Canny(output, 10, 70)      # 邊緣偵測，淺色
    #cv2.imwrite(f'c{cnt}.png', output)
    return output

# 定義一個副程式，從「線」中，找出往上的「延伸線」
def find_line_x(img0, img, y, x, ofs):
    max_up, num, yy, xx = hh - qq, 0, y, x

    for i in range(1, max_up):
        num += 1
        f = 0
        yy = y - i
        for j in range(0, 5):
            if 0 < xx + ofs + j < ww - 1:
                if img[yy, xx + ofs + j] > 127:
                    xx = xx + ofs + j
                    f = 1
                    break
                if img[yy, xx + ofs - j] > 127:
                    xx = xx + ofs - j
                    f = 1
                    break
        if f == 1:
            continue
        if i >= 30:
            break
        return (0, 0, 0)
    return (num, xx, yy)  # num 個點，最後座標 (xx, yy)

# === 設定差速控制 : 左右輪速度控制，以達轉彎 ===
def set_diff_speed(img, speed, steering, stp=0, gain=0.4):
    global l_ofs, r_ofs
    
    delta = int(steering * gain * speed)    # 差速根據方向變化
    l_ofs, r_ofs = -delta, delta            # delta < 0: 右轉, > 0: 左轉
    ofs = abs(delta) - speed
    if ofs > 0:
        l_ofs += ofs
        r_ofs += ofs

    cv2.putText(img, f"({l_ofs},{r_ofs})", (10, 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    #cv2.imwrite(f'd{cnt}.png', img)
    clbrobot.moveforward_ofs(speed, l_ofs, r_ofs, 0.1)

# 定義一個副程式，從合格的「延伸線」中，找出「車道線」
# 修改後的 Wall Following 控制法 (溫柔版)
# 修改後的 Wall Following 控制法 (視覺校正版)
# 修改後的 Wall Following 控制法 (溫柔過彎版)
def get_single_lane_control(img, canny, speed, direction):
    global ww, hh, cnt, lost, auto_go
    
    # --- 1. 硬體校正區 ---
    MOTOR_BIAS = 0.0 

    # --- 2. 目標設定 (維持上一版正確的視覺校正) ---
    if direction == -1: 
        near_target_x = 40
        far_target_x = 40 
    else:               
        near_target_x = 280
        far_target_x = 280 

    # 3. 掃描區域 (修改：擴大近視範圍)
    # 原本 100~240 -> 改成 80~240
    # 讓車子在彎道時，能更久地保持在「溫柔模式」
    y_start_near = 80
    y_end_near = 240
    
    if direction == -1: 
        roi_near = canny[y_start_near:y_end_near, 0:220]
        offset_near = 0 
    else:               
        roi_near = canny[y_start_near:y_end_near, 100:ww]
        offset_near = 100

    white_pixels_near = cv2.findNonZero(roi_near)
    
    final_ex = -1
    final_ey = -1
    is_using_far_view = False 
    found = False

    # 1. 嘗試看近
    if white_pixels_near is not None and len(white_pixels_near) > 10:
        avg_x = np.mean(white_pixels_near[:, 0, 0])
        final_ex = int(avg_x + offset_near)
        final_ey = int((y_start_near + y_end_near) / 2)
        found = True
        current_target = near_target_x
        cv2.circle(img, (final_ex, final_ey), 8, (0, 255, 255), -1) 
    
    else:
        # 2. 近處瞎了，切換看遠 (0~80)
        y_start_far = 0
        y_end_far = 80
        
        if direction == -1: 
            roi_far = canny[y_start_far:y_end_far, 0:220]
            offset_far = 0
        else:               
            roi_far = canny[y_start_far:y_end_far, 100:ww]
            offset_far = 100
            
        white_pixels_far = cv2.findNonZero(roi_far)
        
        if white_pixels_far is not None and len(white_pixels_far) > 5:
            avg_x = np.mean(white_pixels_far[:, 0, 0])
            final_ex = int(avg_x + offset_far)
            final_ey = int((y_start_far + y_end_far) / 2)
            found = True
            is_using_far_view = True 
            current_target = far_target_x 
            
            cv2.circle(img, (final_ex, final_ey), 10, (255, 0, 255), 3)
            print("DEBUG: EMERGENCY! Using Far View!", flush=True)

    # --- 控制邏輯 ---
    if found:
        lost = 0
        cv2.circle(img, (current_target, final_ey), 5, (0, 0, 255), -1)
        
        error_x = current_target - final_ex
        
        if is_using_far_view:
            # 【遠視模式 - 降級】
            # 原本 Kp=0.35, mult=3.0 -> 太暴力導致自轉
            # 改回跟近視差不多，只稍微強一點點
            Kp = 0.25      
            turn_mult = 2.5 
            txt_mode = "FAR"
            cte = error_x * Kp 
        else:
            # 【近視模式】
            Kp = 0.20      
            turn_mult = 2.0 
            txt_mode = "NEAR"
            cte = error_x * Kp
            if abs(cte) < 2: cte = 0 

        cte = int(cte)
        
        base_speed = 40 
        
        turn_power = cte * turn_mult
        
        # 限制最大轉向力道，防止單輪倒轉過快導致甩尾
        # 如果 turn_power 太大 (例如 > 40)，會導致一邊輪子變成負數(倒轉)
        # 我們限制一下，讓它順順轉
        if turn_power > 50: turn_power = 50
        if turn_power < -50: turn_power = -50

        # 馬達 Bias 計算
        l_base = base_speed
        r_base = base_speed
        l_base = l_base * (1.0 - MOTOR_BIAS)
        r_base = r_base * (1.0 + MOTOR_BIAS)

        l_speed = l_base - turn_power
        r_speed = r_base + turn_power
        
        l_speed = max(min(l_speed, 100), -100)
        r_speed = max(min(r_speed, 100), -100)
        
        cv2.putText(img, f"{txt_mode} Err:{int(error_x)}", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        clbrobot.MotorRun(0, 'forward', int(l_speed))
        clbrobot.MotorRun(1, 'forward', int(r_speed))
        clbrobot.MotorRun(2, 'forward', int(l_speed))
        clbrobot.MotorRun(3, 'forward', int(r_speed))
        
        return img

    else:
        if lost < 20:
            lost += 1
            print(f"*** Lost count {lost}...", flush=True)
            clbrobot.moveforward(35, 0)
            return img
        else:
            print("Lost too long. Stop.")
            auto_go = 0
            clbrobot.t_stop(0.1)

    return img


def capture_loop():                 # 另一個任務的執行緒
    global picamera, latest_frame, running
    global car_go, speed, l_ofs, r_ofs
    target_fps = 20
    interval = 1.0 / target_fps
    while running:
        frame = picamera.capture_array()  # RGB
        with frame_lock:            # 只能有一個 thread 存取 latest_frame
            latest_frame = frame
                        
        # 車子遙控前進
        if car_go == 1 and auto_go == 0:
            with speed_lock:
                clbrobot.moveforward_ofs(speed, l_ofs, r_ofs, 0.1)

        time.sleep(interval)

t = threading.Thread(target=capture_loop, daemon=True)  # capture_loop 執行緒
t.start()

# ------------------------------------------------------------------
# ------------------------- Flask 初始化 ----------------------------        
app = Flask(__name__)

def live_mjpeg():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
            
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

# --- 新增變數：紀錄目前追蹤的是左線(-1)還是右線(1)，預設先看右線 ---
tracking_dir = 1  
lost_counter = 0  # 迷路計數器

# --- 新增函式：掃描該方向是否有車道線，並返回分數 (找到的點數) ---
def get_lane_score(canny, direction):
    global ww, hh
    w_mid = int(ww / 2)
    
    # 修改：不再只看下半部，改為搜尋「整個垂直區域」
    # 這樣就算線只出現在角落（上方），也能被算進分數裡
    if direction == 1: 
        # 看右邊 (整個右半部)
        roi = canny[0:hh, w_mid:ww]
    else: 
        # 看左邊 (整個左半部)
        roi = canny[0:hh, 0:w_mid]
        
    score = cv2.countNonZero(roi)
    return score
    
switch_confirm_count = 0 

def live_mjpeg_processed():
    global tracking_dir, lost_counter, auto_go, speed, switch_confirm_count
    
    print("DEBUG: Stream started", flush=True)

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        canny = get_canny(frame)
        
        if auto_go == 1:
            score_left = get_lane_score(canny, -1)
            score_right = get_lane_score(canny, 1)
            
            # --- 防手震切換邏輯 ---
            
            # 1. 基本門檻
            HYSTERESIS = 150 
            WEAK_LINE_THRESHOLD = 150
            
            # 判斷是否「想要」切換
            want_switch = False
            new_dir = tracking_dir
            
            # 正在追左線 (-1) -> 想切右?
            if tracking_dir == -1:
                if score_right > score_left + HYSTERESIS and score_left < WEAK_LINE_THRESHOLD:
                    want_switch = True
                    new_dir = 1
            
            # 正在追右線 (1) -> 想切左?
            elif tracking_dir == 1:
                if score_left > score_right + HYSTERESIS and score_right < WEAK_LINE_THRESHOLD:
                    want_switch = True
                    new_dir = -1
            
            # --- 2. 確認機制 (Debounce) ---
            if want_switch:
                switch_confirm_count += 1
                # print(f"DEBUG: Want switch... {switch_confirm_count}/5", flush=True) # 除錯用
            else:
                switch_confirm_count = 0 # 如果中間有一幀不滿足，就重置計數
            
            # 3. 只有連續 5 幀 (約0.2秒) 都想切換，才真的執行
            if switch_confirm_count >= 3:
                print(f"DEBUG: CONFIRMED Switch to {'RIGHT' if new_dir==1 else 'LEFT'}! (L={score_left}, R={score_right})", flush=True)
                tracking_dir = new_dir
                lost_counter = 0
                switch_confirm_count = 0 # 重置
                
                # 切換瞬間的慣性處理 (保留之前的暴力轉向，幫助過彎)
                if tracking_dir == 1: # 切右
                    clbrobot.MotorRun(0, 'forward', 80)
                    clbrobot.MotorRun(1, 'forward', -20)
                    clbrobot.MotorRun(2, 'forward', 80)
                    clbrobot.MotorRun(3, 'forward', -20)
                else: # 切左
                    clbrobot.MotorRun(0, 'forward', -20)
                    clbrobot.MotorRun(1, 'forward', 80)
                    clbrobot.MotorRun(2, 'forward', -20)
                    clbrobot.MotorRun(3, 'forward', 80)
                time.sleep(0.1)

            # 4. 取得目前分數 & 開車
            if tracking_dir == -1: score_current = score_left
            else:                  score_current = score_right
            
            print(f"DEBUG: Dir={tracking_dir}, Score={score_current}", flush=True)

            if score_current > 100:
                # 呼叫控制函式 (請確認這裡用的是 Target 110/210 的版本)
                frame = get_single_lane_control(frame, canny, speed, direction=tracking_dir)
                lost_counter = 0  
                
                txt = "Right Lane" if tracking_dir == 1 else "Left Lane"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                lost_counter += 1
                print(f"DEBUG: LOST ALL! Count={lost_counter}", flush=True)
                if lost_counter < 10:
                     clbrobot.moveforward(35, 0)
                else:
                    print("DEBUG: STOPPING CAR.", flush=True)
                    auto_go = 0
                    clbrobot.t_stop(0.1)
        else:
            frame = canny

        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")
        
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)
    #return render_template('index.html')

@app.route("/live")
def video_feed():
    return Response(live_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live2")
def video_feed2():
    return Response(live_mjpeg_processed(), 
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/turn_left", methods=["POST"])
def turn_left():
    global angle
    with angle_lock:
        angle = min(MAX_ANGLE, angle + STEP_ANGLE)      # 鏡頭左轉
    clbrobot.set_servo_angle(Cam_X, angle, 0.1)
    return jsonify({"ok": True, "angle": angle})

@app.route("/api/turn_right", methods=["POST"])
def turn_right():
    global angle
    with angle_lock:
        angle = max(MIN_ANGLE, angle - STEP_ANGLE)      # 鏡頭右轉
    clbrobot.set_servo_angle(Cam_X, angle, 0.1)
    return jsonify({"ok": True, "angle": angle})

@app.route("/api/turn_up", methods=["POST"])
def turn_up():
    global updown
    with angle_lock:
        updown = max(MIN_UPDOWN, updown - STEP_ANGLE/2)
    clbrobot.set_servo_angle(Cam_Y, updown, 0.1)
    return jsonify({"ok": True, "updown": updown})

@app.route("/api/turn_down", methods=["POST"])
def turn_down():
    global updown
    with angle_lock:
        updown = min(MAX_UPDOWN, updown + STEP_ANGLE/2)
    clbrobot.set_servo_angle(Cam_Y, updown, 0.1)
    return jsonify({"ok": True, "updown": updown})

@app.route("/api/accelerate", methods=["POST"])
def accelerate():
    global car_go, speed, l_ofs, r_ofs
    with speed_lock:
        if car_go == 1:
            speed = min(speed + SPEED_STEP, SPEED_MAX)  # 車子加速前進
            l_ofs, r_ofs = 0, 0
    return jsonify({"ok": True, "speed": speed, "go": True})

@app.route("/api/decelerate", methods=["POST"])
def decelerate():
    global car_go, speed, l_ofs, r_ofs
    with speed_lock:
        if car_go == 1:
            speed = max(speed - SPEED_STEP, SPEED_MIN)  # 車子減速前進
            l_ofs, r_ofs = 0, 0
    return jsonify({"ok": True, "speed": speed, "go": True})

@app.route("/api/left_forward", methods=["POST"])
def left_forward():
    global car_go, speed, l_ofs, r_ofs
    # 左轉：左輪減速、右輪加速（若方向相反，對調正負號即可）
    with speed_lock:
        if car_go == 1:
            l_ofs, r_ofs = l_ofs-1, 0                        # 車子前進左轉
    return jsonify({"ok": True, "speed": speed, "go": True})

@app.route("/api/right_forward", methods=["POST"])
def right_forward():
    global car_go, speed, l_ofs, r_ofs
    with speed_lock:
        if car_go == 1:
            l_ofs, r_ofs = 0, r_ofs-1                        # 車子前進右轉
    return jsonify({"ok": True, "speed": speed, "go": True})

@app.route("/api/start", methods=["POST"])
def start():
    global car_go, speed, l_ofs, r_ofs, auto_go
    with speed_lock:
        l_ofs, r_ofs = 0, 0
        car_go, auto_go = 1, 0
    return jsonify({"ok": True, "speed": speed, "go": True})

@app.route("/api/stop", methods=["POST"])
def stop():
    global car_go, speed, l_ofs, r_ofs, auto_go
    with speed_lock:
        car_go, auto_go = 0, 0
        clbrobot.t_stop(0.1)
    return jsonify({"ok": True, "speed": speed, "go": False})

@app.route("/api/autogo", methods=["POST"])
def autogo():
    global car_go, speed, l_ofs, r_ofs, lost, auto_go
    with speed_lock:
        l_ofs, r_ofs, lost = 0, 0, 0
        car_go, auto_go = 0, 1
        # 刪除或註解掉下面這兩行
        #clbrobot.stop_servo_angle(Cam_X)
        #clbrobot.stop_servo_angle(Cam_Y)
    return jsonify({"ok": True, "speed": speed, "go": True})

def cleanup():
    global running
    running = False
    clbrobot.t_stop(0.1)

    try:
        t.join(timeout=1.0)
    except Exception:
        pass
    try:
        picamera.stop()
    except Exception:
        pass

if __name__ == "__main__":
    
    clbrobot.set_servo_angle(Cam_X, angle, 1)   # 開機置中
    clbrobot.set_servo_angle(Cam_Y, updown, 1)   # 開機置中

    try:
        # threaded=True 讓影像串流與 API 控制並行
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    finally:
        cleanup()
    
