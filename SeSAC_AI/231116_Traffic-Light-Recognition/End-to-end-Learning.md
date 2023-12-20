# End-to-end Learning for Inter-Vehicle Distance and Relative Velocity
# Estimation in ADAS with a Monocular Camera

- paper: [https://arxiv.org/pdf/2006.04082v2.pdf](https://arxiv.org/pdf/2006.04082v2.pdf)
- github: [https://github.com/ZhenboSong/mono_velocity](https://github.com/ZhenboSong/mono_velocity

## 목표와 도전 과제

- End-to-end 학습을 기반으로 1개의 단안 카메라 기반 차량 간 거리 및 상대 속도 추정
- 2개의 연속 프레임을 바탕으로 원근 왜곡의 영향을 완화하기 위한 차량 중심 샘플링 메커니즘 제안


## 방법론

- 도로 표면과 카메라 높이 사이의 기하학적 제약을 활용하여 거리 추정의 스케일 모호성 해결
- 2D 바운딩 박스가 3D 바운딩 박스 투영을 둘러싸고 있다고 가정하여 차량 거리 회귀 모델 제안
	- 감지된 2D 바운딩 박스의 균일한 폭과 높이를 입력값으로 사용
- 상대 속도는 시간 간격 동안 측정 카메라에 관측된 차량의 움직으로 추정
    - 속도가 빠르더라도 먼 거리에 있는 차량은 이미지의 차이가 작음
	- 속도가 느리더라도 근접 차량은 이미지의 차이가 큼

## 모델 아키텍처

- Monocular (이하 단안) depth와 3D 객체 검출 방법을 적용
- U-net 구조를 도입하여 지도 학습에 의해 깊이를 예측
- DORN 은 깊이를 순서형 회귀 문제로 간주
- M3D-RPN 은 3D 바운딩 박스가 2D 이미지 공간에서 생성된 Convolution 특징을 활용할 수 있는 3D 영역을 제안
- 상대속도는 이미지의 흐름을 3D 모션 필드로 나타냄
	- Flownet2, PWC-Net을 통해 여러 신경망을 stack, warping
	- 이를 통해, 여러 신경망을 통합하여 가볍고 빠른 신경망 구현

![Screenshot 2023-12-20 at 2 57 16 PM](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/33dce05b-d1f6-4bf7-b442-88e025baf8d0)

<img width="1005" alt="Screenshot 2023-12-20 at 4 52 22 PM" src="https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/38f7bb53-64eb-43d3-842a-8553bc79dc22">

## 데이터셋과 사전 훈련

- Tusimple velocity Dataset : 20 fps, 40 frame 비디오 시퀀
- KITTI Dataset
- FlyingChairs에 사전 훈련된 PWCNet을 기반으로 Feature map과 flow map을 통합하기 위해 7x7 ROI Align 사용
- Conv layer size : 3x3, 7x7
- Resize 284 x 448
- concatenating the geometric vector, deep feature vector and flow vector
- 4 fully connected layers with ReLU activation function are employed to compute the distance and velocity
- The whole network is implemented in PyTorch and trained end-to-end by optimizing the loss function with ADAM


## Loss Function and Evaluation Metrics

1. Absolute Relative Error (REL): 
$$
\quad \text{REL} = \frac{1}{M} \sum_{i=1}^{M} \left| \frac{d_i - \hat{d}_i}{d_i} \right|
$$
2. Root Mean Squared Error (RMSE):
$$
\quad \text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \left| d_i - \hat{d}_i \right|^2}
$$

3. Average Log10 Error: 
$$ 
\quad \text{Average Log10 Error} = \frac{1}{M} \sum_{i=1}^{M} \left| \log_{10}(d_i) - \log_{10}(\hat{d}_i) \right| 
$$

4. Threshold Accuracy $(\delta^n)$:
$$
\begin{align*}
\text{Percentage of pixels where} \quad \max \left( \frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i} \right) < 1.25^n \quad \text{for } n = 1, 2, 3 \\
\delta^n &: \text{Threshold Accuracy for } n = 1, 2, 3 \\
d_i &: \text{Ground Truth Depth at pixel } i \\
\hat{d}_i &: \text{Predicted Depth at pixel } i \\
M &: \text{Total Number of Pixels in the Image}
\end{align*}
$$

5. Mean Relative Improvement across Datasets (mRID):
$$
\quad \text{mRID} = \frac{1}{M} \sum_{i=1}^{M} \text{RID}_i
$$
6. Mean Relative Improvement across Metrics (mRI$\theta$)
$$
\quad \text{mRI}\theta = \frac{1}{N} \sum_{j=1}^{N} \text{RI}\theta_j
$$

7. Relative Improvement (RI) for lower-is-better metrics:
$$
 \quad \text{RI} = \frac{r - t}{r}
$$


8. Relative Improvement (RI) for higher-is-better metrics:
$$ \quad \text{RI} = \frac{t - r}{r} $$
$$
\\ r: \text{Reference Score} \\ t: \text{Target Score} $$


- ZoeDepth는 scale-invariant log loss를 사용하여 깊이 추정의 정확도 측정
	- 이 loss function은 깊이 추정에서의 스케일 불변성을 보장하여, 다양한 크기의 객체에 대한 깊이 추정을 일관되게 수행할 수 있도록 함
- 모델의 성능 평가에는 정확도, 정밀도, 재현율과 같은 표준 메트릭스가 사용



## 결과




