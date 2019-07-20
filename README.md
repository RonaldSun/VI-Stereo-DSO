# VI-Stereo-DSO

## Related Papers:
- **Direct Sparse Odometry**, *J. Engel, V. Koltun, D. Cremers*, In arXiv:1607.02565, 2016

- **Stereo DSO:Large-Scale Direct Sparse Visual Odometry with Stereo Cameras**, *Rui Wang, Martin Schwörer, Daniel Cremers*, 2017 IEEE International Conference on Computer Vision

- **Direct Sparse Visual-Inertial Odometry using Dynamic Marginalization**, *Lukas von Stumberg, Vladyslav Usenko, Daniel Cremers*, 2018 IEEE International Conference on Robotics and Automation (ICRA)

## 1. Installation
Please follow https://github.com/JakobEngel/dso.

## 2. Usage
1. Modify the bash file

2. For stereo:

    ```
    bash ./run.bash
    ```

3. For mono:

    ```
    bash ./run_mono.bash
    ```

- T_stereo file: 

```
    transform matrix from right camera to left camera (4x4).
```
- imu_info file: 
```
    transform matrix from left camera to imu (4x4).
    gyroscope noise density [ rad / s / sqrt(Hz) ] ( gyro "white noise" )
    accelerometer noise density [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
    gyroscope random walk [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
    accelerometer random walk [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion ) 
```
## Experiments

- 20190424

SE(3) Umeyama alignment：

| weight 6,0.6,0.5 | MH01 | MH02 | MH03 | MH04 | MH05  | V101 | V102 | V103 | V201 | V202 | V203 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| ape rmse(m)| 0.0321 | 0.0288 | 0.0743 | 0.119 | 0.072 | 0.0599 | 0.105 | 0.168 | 0.0852 | 0.0667 | 0.211 |

![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/MH03_1.png)
![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/MH04_1.png)
![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/MH05_1.png)
![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/V102_1.png)



No alignment（Initialization may have an impact on the results）:

| weight 6,0.6,0.5 | MH01 | MH02 | MH03 | MH04 | MH05  | V101 | V102 | V103 | V201 | V202 | V203 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| ape rmse(m)| 0.0993 | 0.0557 | 0.234 | 0.194 | 0.111 | 0.232 |0.202 | 0.2777 | 0.102 | 0.114 | 0.263|

- 20190416

Apply First Estimates Jacobians to scale to maintain consistency.

EuRoC MH01:

![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/MH01.png)

- EuRoC V1_03:

![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/euroc_v1_03.png)

- EuRoC V2_03:

![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/euroc_V2_03.png)

- 20190409:

![](https://github.com/RonaldSun/VI-Stereo-DSO/blob/master/pic/2019-04-09-V203.png)

green line: groundtruth, redline: VI-Stereo-DSO

## P.S.

This project aims at verifying the location performance of VI-Stero-DSO. The computational efficiency is to be improved. Actually in current code, many preintergration operations are redundant and can be simplified.