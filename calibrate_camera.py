import cv2
import numpy as np
import glob

# チェスボードの交点数
CHECKERBOARD = (12, 7)

# 3Dポイントと2Dポイントを格納するリスト
objpoints = []
imgpoints = []

# 3Dポイントの準備（Z=0の平面上）
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 画像の読み込み
images = glob.glob('calib_images_B/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

# カメラキャリブレーション
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# 結果表示
print("Camera Matrix (Intrinsic parameters):\n", mtx)
print("\nDistortion Coefficients:\n", dist)

# 再投影誤差の計算
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("\nAverage re-projection error: ", total_error / len(objpoints))