# head-pose-estimation
Real-time head pose estimation built with OpenCV and dlib 

<b>2D:</b><br>Using dlib for facial features tracking, modified from http://dlib.net/webcam_face_pose_ex.cpp.html
<br>The algorithm behind it is described in http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf
<br>It applies cascaded regression trees to predict shape(feature locations) change in every frame.
<br>Splitting nodes of trees are trained in random, greedy, maximizing variance reduction fashion.
<br>The well trained model can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
<br>Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)
<br><br>
<b>3D:</b><br>To match with 2D image points(facial features) we need their corresponding 3D model points. 
<br>http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp provides a similar 3D facial feature model.
<br>It's annotation is shown below:<br><br>
![gl](https://cloud.githubusercontent.com/assets/16308037/24229340/ea8bad94-0fb3-11e7-9e1d-0a2217588ba4.jpg)
<br><br>
Finally, with solvepnp function in OpenCV, we can achieve real-time head pose estimation.
<br><br>

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




---------------

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



