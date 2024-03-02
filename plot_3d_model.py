"""
# 顔追跡とヘッドポーズ推定

このPythonスクリプトは、リアルタイムでの顔追跡とヘッドポーズ推定を行うためのものです。OpenCV、dlib、imutils、numpyなどのライブラリを使用しています。顔のランドマーク検出にはdlibの68点ランドマークモデルを使用し、得られたランドマークを元に頭部の姿勢を推定します。

## 必要条件

- Python 3.x
- OpenCV
- dlib
- imutils
- numpy

## 特徴

- **カメラキャリブレーション**: スクリプトは、カメラの内部パラメータと歪み係数を使用して、より正確なヘッドポーズ推定を行います。
- **移動平均フィルタ**: 各ランドマーク点に対して、過去5フレームの移動平均を取ることで、ジッターを減らし、滑らかな動きを実現します。
- **ヘッドポーズ推定**: 顔のランドマークを基に、頭部の姿勢をロール、ピッチ、ヨーで推定します。
- **リアルタイム顔追跡**: ビデオフレームから顔を検出し、ランドマークを描画します。また、顔の向きに応じて3Dで立方体を描画することで、ヘッドポーズを視覚的に表現します。

## 使用方法

1. 必要なライブラリをインストールします。
2. `shape_predictor_68_face_landmarks.dat` ファイルをプロジェクトのルートディレクトリに配置します（dlibの顔のランドマーク検出モデル）。
3. スクリプトを実行します: `python <スクリプト名>.py`

カメラが開始され、リアルタイムで顔を追跡し、ヘッドポーズを推定します。`q`キーを押すと、プログラムが終了します。

## 注意点

- カメラのキャリブレーションパラメータ（`K`と`D`）は、使用するカメラに応じて調整する必要があります。
- 室内光などの照明条件によっては、顔検出の精度が影響を受ける場合があります。
- ヘッドポーズの精度は、顔のランドマーク検出の精度に依存します。

このスクリプトは教育目的で作成されており、商用利用や高度な要件には追加の調整が必要になる場合があります。

"""


"""
# Face Tracking and Head Pose Estimation

This Python script is designed for real-time face tracking and head pose estimation using libraries such as OpenCV, dlib, imutils, and numpy. It utilizes dlib's 68-point landmark model for facial landmark detection and estimates the head pose based on these landmarks.

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- numpy

## Features

- **Camera Calibration**: The script uses camera intrinsic parameters and distortion coefficients for more accurate head pose estimation.
- **Moving Average Filter**: A moving average of the last 5 frames for each landmark point is calculated to reduce jitter and achieve smooth motion.
- **Head Pose Estimation**: The head pose is estimated in terms of roll, pitch, and yaw angles based on facial landmarks.
- **Real-time Face Tracking**: Faces are detected from video frames, landmarks are plotted, and a cube is drawn in 3D according to the orientation of the head to visually represent the head pose.

## Usage

1. Install the required libraries.
2. Place the `shape_predictor_68_face_landmarks.dat` file in the root directory of the project (dlib's facial landmark detection model).
3. Run the script: `python <script_name>.py`

The camera will start, and faces will be tracked in real-time with head pose estimation. Press the `q` key to exit the program.

## Notes

- The camera calibration parameters (`K` and `D`) need to be adjusted according to the camera being used.
- Lighting conditions, such as indoor lighting, may affect the accuracy of face detection.
- The accuracy of the head pose is dependent on the accuracy of facial landmark detection.

This script is created for educational purposes and may require additional adjustments for commercial use or advanced requirements.

"""

import cv2
import dlib
import numpy as np
from imutils import face_utils
from collections import deque

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

# カメラキャリブレーションパラメータ
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([
    [6.825897, 6.760612, 4.402142],
    [1.330353, 7.122144, 6.903745],
    [-1.330353, 7.122144, 6.903745],
    [-6.825897, 6.760612, 4.402142],
    [5.311432, 5.485328, 3.987654],
    [1.789930, 5.393625, 4.413414],
    [-1.789930, 5.393625, 4.413414],
    [-5.311432, 5.485328, 3.987654],
    [2.005628, 1.409845, 6.165652],
    [-2.005628, 1.409845, 6.165652],
    [2.774015, -2.080775, 5.048531],
    [-2.774015, -2.080775, 5.048531],
    [0.000000, -3.116408, 6.097667],
    [0.000000, -7.415691, 4.070434]
])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# 再投影のためのソースポイント
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# 移動平均フィルタの設定
points_history = {i: deque(maxlen=5) for i in range(68)}  # 各点に対して5点分の履歴を保持

def apply_moving_average(shape):
    averaged_shape = np.zeros(shape.shape, dtype=np.float32)
    for i, point in enumerate(shape):
        points_history[i].appendleft(point)
        averaged_shape[i] = np.mean(points_history[i], axis=0)
    return averaged_shape

def get_head_pose(shape, object_pts, cam_matrix, dist_coeffs):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(lambda point: (int(point[0]), int(point[1])), reprojectdst.reshape(8, 2)))

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle, rotation_vec, translation_vec

def draw_cube(frame, points):
    # 立方体の各面を定義
    faces = [
        [points[0], points[1], points[2], points[3]], # 前面
        [points[4], points[5], points[6], points[7]], # 背面
        [points[0], points[1], points[5], points[4]], # 上面
        [points[2], points[3], points[7], points[6]], # 下面
        [points[0], points[3], points[7], points[4]], # 左面
        [points[1], points[2], points[6], points[5]]  # 右面
    ]

    # 各面を塗りつぶす
    for face in faces:
        cv2.fillConvexPoly(frame, np.array(face, dtype=np.int32), (0, 255, 0))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)
            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                
                # 移動平均フィルタを適用
                averaged_shape = apply_moving_average(shape)

                reprojectdst, euler_angle, rotation_vec, translation_vec = get_head_pose(averaged_shape, object_pts, cam_matrix, dist_coeffs)

                draw_cube(frame, reprojectdst)
                
                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 50, 0),3)


                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()
