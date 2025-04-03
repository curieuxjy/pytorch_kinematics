> [!NOTE]  
> README.md 한국어 번역

# PyTorch 로봇 운동학
- 병렬 및 미분 가능한 순방향 운동학(FK), Jacobian 계산, 감쇠 최소자승법(damped least squares)을 통한 역운동학(IK)
- URDF, SDF, MJCF 형식의 로봇 설명 파일 불러오기 지원
- [pytorch-volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric)을 통해 여러 로봇 구성 및 쿼리 포인트에 대해 SDF query 병렬 처리 가능

# 설치 방법
```shell
pip install pytorch-kinematics
```

개발 모드에서는 저장소를 클론한 뒤, `pip3 install -e .` 명령어로 수정 가능한 상태로 설치합니다.

> [!TIP]  
> (`pyproject.toml` 파일에 `requires-python = ">=3.6"`라고 명시. Python 3.10으로 테스트. )
> - `pip install mujoco`: `build_chain_from_mjcf` 함수 이용시 필요 (`src/pytorch_kinematics/__init__.py`)
> - `tests/mujoco_menagerie`: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 소스 가져오기
> - `pip install pybullet`: `tests/test_inverse_kinematics.py` 실행시 필요
> - `tests/` 폴더 내의 모든 python script 확인: `for file in *.py; do python "$file"; done`

## Reference
[![DOI](https://zenodo.org/badge/331721571.svg)](https://zenodo.org/badge/latestdoi/331721571)

이 패키지를 연구에 사용한 경우 다음과 같이 인용해 주세요.
```
@software{Zhong_PyTorch_Kinematics_2024,
author = {Zhong, Sheng and Power, Thomas and Gupta, Ashwin and Mitrano, Peter},
doi = {10.5281/zenodo.7700587},
month = feb,
title = {{PyTorch Kinematics}},
version = {v0.7.1},
year = {2024}
}
```

# 사용법

코드 예제는 `tests` 디렉토리와 아래 일부 샘플을 참고하세요.

## 로봇 불러오기
```python
import pytorch_kinematics as pk

urdf = "widowx/wx250s.urdf"
# there are multiple natural end effector links so it's not a serial chain
chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
# visualize the frames (the string is also returned)
chain.print_tree()
"""
base_link
└── shoulder_link
    └── upper_arm_link
        └── upper_forearm_link
            └── lower_forearm_link
                └── wrist_link
                    └── gripper_link
                        └── ee_arm_link
                            ├── gripper_prop_link
                            └── gripper_bar_link
                                └── fingers_link
                                    ├── left_finger_link
                                    ├── right_finger_link
                                    └── ee_gripper_link
"""

# extract a specific serial chain such as for inverse kinematics
serial_chain = pk.SerialChain(chain, "ee_gripper_link", "base_link")
serial_chain.print_tree()
"""
base_link
└── shoulder_link
    └── upper_arm_link
        └── upper_forearm_link
            └── lower_forearm_link
                └── wrist_link
                    └── gripper_link
                        └── ee_arm_link
                            └── gripper_bar_link
                                └── fingers_link
                                    └── ee_gripper_link
"""

# you can also extract a serial chain with a different root than the original chain
serial_chain = pk.SerialChain(chain, "ee_gripper_link", "gripper_link")
serial_chain.print_tree()
"""
 gripper_link
└── ee_arm_link
    └── gripper_bar_link
        └── fingers_link
            └── ee_gripper_link
"""
```

## 순방향 운동학(FK)
```python
import math
import pytorch_kinematics as pk

# load robot description from URDF and specify end effector link
# URDF 파일에서 로봇 설명을 불러오고, 엔드 이펙터 링크를 지정합니다.
chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")

# prints out the (nested) tree of links
# 링크들의 계층 트리 구조를 출력합니다.
print(chain)

# prints out list of joint names
# 관절 이름 리스트를 출력합니다.
print(chain.get_joint_parameter_names())

# specify joint values (can do so in many forms)
# 관절 값들을 지정합니다 (여러 가지 형식으로 지정 가능).
th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]

# do forward kinematics and get transform objects;
# end_only=False gives a dictionary of transforms for all links
# 순방향 운동학을 수행하고 변환 객체(transform)를 얻습니다;
# end_only=False이면 모든 링크에 대한 변환 정보를 딕셔너리 형태로 반환합니다.
ret = chain.forward_kinematics(th, end_only=False)

# look up the transform for a specific link
# 특정 링크("lbr_iiwa_link_7")에 대한 변환을 조회합니다.
tg = ret['lbr_iiwa_link_7']

# get transform matrix (1,4,4), then convert to separate position and unit quaternion
# 변환 행렬 (1,4,4)을 가져오고, 이를 위치(pos)와 단위 쿼터니언(rot)으로 분리합니다.
m = tg.get_matrix()
pos = m[:, :3, 3]
rot = pk.matrix_to_quaternion(m[:, :3, :3])
```

2D 조인트 값을 입력하여 FK를 병렬화할 수 있으며, 가능하다면 CUDA도 사용할 수 있습니다.

```python
import torch
import pytorch_kinematics as pk

# CUDA 사용 가능 여부에 따라 디바이스 설정
d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64  # 더 높은 정밀도의 계산을 위해 float64 사용

# URDF 파일로부터 직렬 체인 로봇을 생성하고, 디바이스 및 데이터 타입 설정
chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
chain = chain.to(dtype=dtype, device=d)

# N개의 샘플을 생성
N = 1000
# 관절 개수에 맞게 무작위 관절값(batch)을 생성 (크기: N x 관절수)
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)

# order of magnitudes faster when doing FK in parallel
# 병렬로 순방향 운동학(FK)을 계산하면 훨씬 빠릅니다 (성능 향상은 수십~수백 배).
# elapsed 0.008678913116455078s for N=1000 when parallel
# 병렬 계산 시 N=1000개의 관절 구성을 약 0.009초만에 처리함
# (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
# 반환값은 (N, 4, 4) 크기의 변환 행렬이며, 기본적으로 end_effector만 포함됩니다 (end_only=True).
tg_batch = chain.forward_kinematics(th_batch)

# elapsed 8.44686508178711s for N=1000 when serial
# 순차적으로 FK 계산 시에는 약 8.4초가 소요됨 (병렬보다 훨씬 느림)
for i in range(N):
    tg = chain.forward_kinematics(th_batch[i])  # 각 샘플에 대해 순차적으로 FK 수행
```

FK를 통해 기울기(gradient)를 계산할 수 있습니다

```python
import torch
import math
import pytorch_kinematics as pk

# URDF 파일을 통해 직렬 체인을 구성하고, 엔드 이펙터를 지정
chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")

# require gradient through the input joint values
# 입력된 관절 값에 대해 미분(gradient)이 가능하도록 설정
th = torch.tensor(
    [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], 
    requires_grad=True  # 역전파를 통해 gradient 계산 가능하게 함
)

# 순방향 운동학 계산 (입력된 관절값을 기반으로 엔드 이펙터의 위치 및 자세 계산)
tg = chain.forward_kinematics(th)

# 변환 행렬 (1, 4, 4) 추출
m = tg.get_matrix()

# 위치 벡터 추출: (x, y, z) 좌표 (마지막 열의 앞 3개 요소)
pos = m[:, :3, 3]

# 위치 벡터의 norm (거리) 에 대해 역전파 수행
# pos.norm()은 스칼라이며, backward()를 호출하면 입력 th에 대한 gradient가 계산됨
pos.norm().backward()

# now th.grad is populated
# 이제 th.grad에는 각 관절 값에 대한 위치 노름의 gradient가 저장됨
```

SDF 및 MJCF 형식의 파일도 로드할 수 있으며, serial chain 구조가 아닌 경우 dictionary를 통해 조인트 값을 전달할 수 있습니다
(지정되지 않은 조인트는 th=0을 얻습니다)

```python
import math
import torch
import pytorch_kinematics as pk

# SDF 파일로부터 로봇 체인을 구성
chain = pk.build_chain_from_sdf(open("simple_arm.sdf").read())

# 관절 값을 딕셔너리 형태로 지정하여 순방향 운동학 수행
ret = chain.forward_kinematics({
    'arm_elbow_pan_joint': math.pi / 2.0, 
    'arm_wrist_lift_joint': -0.5
})

# recall that we specify joint values and get link transforms
# 관절 값을 지정하면, 링크에 대한 변환(Transform)을 얻습니다
tg = ret['arm_wrist_roll']  # 해당 링크의 변환 결과 추출

# can also do this in parallel
# 이 작업은 병렬로도 수행 가능합니다
N = 100
ret = chain.forward_kinematics({
    'arm_elbow_pan_joint': torch.rand(N, 1), 
    'arm_wrist_lift_joint': torch.rand(N, 1)
})
# (N, 4, 4) transform object
# 출력은 (N, 4, 4) 크기의 변환 행렬 (N은 배치 크기)
tg = ret['arm_wrist_roll']

# building the robot from a MJCF file
# MJCF 파일을 통해 로봇 체인을 생성
chain = pk.build_chain_from_mjcf(open("ant.xml").read())

# 로봇 트리 구조와 관절 이름 출력
print(chain)
print(chain.get_joint_parameter_names())

# 관절 값을 딕셔너리로 지정하고 순방향 운동학 계산
th = {'hip_1': 1.0, 'ankle_1': 1}
ret = chain.forward_kinematics(th)

# 또 다른 MJCF 예시 (휴머노이드)
chain = pk.build_chain_from_mjcf(open("humanoid.xml").read())
print(chain)
print(chain.get_joint_parameter_names())

# 간단한 관절 설정으로 순방향 운동학 수행
th = {'left_knee': 0.0, 'right_knee': 0.0}
ret = chain.forward_kinematics(th)
```


## Jacobian 계산
Jacobian (운동학적(kinematics) 맥락에서)은 관절 값 변화에 따라 엔드 이펙터가 어떻게 변하는지를 설명하는 행렬입니다.
(![dx](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdot%7Bx%7D)는 twist, 또는 stacked velocity 와 angular velocity):
![jacobian](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdot%7Bx%7D%3DJ%5Cdot%7Bq%7D) 

`SerialChain`의 경우, base frame에 대한 Jacobian 계산을 위한 **미분 가능하고 병렬화 가능한** 방법을 제공합니다.

```python
import math
import torch
import pytorch_kinematics as pk

# can convert Chain to SerialChain by choosing end effector frame
# 체인(Chain)을 직렬 체인(SerialChain)으로 변환할 수 있으며, 이때 엔드 이펙터 링크를 지정합니다
chain = pk.build_chain_from_sdf(open("simple_arm.sdf").read())

# print(chain) to see the available links for use as end effector
# print(chain)를 사용하여 어떤 링크들이 있는지 확인 가능

# note that any link can be chosen; it doesn't have to be a link with no children
# 꼭 자식 링크가 없는 끝단 링크일 필요는 없으며, 어느 링크든 엔드 이펙터로 선택할 수 있음
chain = pk.SerialChain(chain, "arm_wrist_roll_frame")

# URDF로부터 직렬 체인 생성 (KUKA iiwa 7-DOF)
chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")

# 관절 값 입력 (7개의 자유도)
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])

# (1, 6, 7) 크기의 텐서: 로봇의 7개 관절에 대한 Jacobian 계산
# 6은 트위스트(twist)의 자유도: 3개 선형 속도 + 3개 각속도
J = chain.jacobian(th)

# get Jacobian in parallel and use CUDA if available
# Jacobian 계산을 병렬로 수행하며, CUDA가 가능하면 GPU를 사용함
N = 1000  # 배치 사이즈
d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64  # 더 정밀한 계산

# 체인을 디바이스와 데이터 타입에 맞게 옮김
chain = chain.to(dtype=dtype, device=d)

# Jacobian calculation is differentiable
# Jacobian 계산은 autograd를 통해 미분 가능함

# 관절 값을 무작위로 N개 생성, gradient 추적 활성화
th = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)

# (N, 6, 7): 배치로 Jacobian 계산
J = chain.jacobian(th)

# can get Jacobian at a point offset from the end effector (location is specified in EE link frame)
# 엔드 이펙터 기준으로 오프셋 위치에 대해 Jacobian을 계산할 수도 있음
# by default location is at the origin of the EE frame
# 기본 위치는 EE 프레임의 원점

# 각 배치마다 엔드 이펙터로부터의 상대 위치를 무작위로 설정
loc = torch.rand(N, 3, dtype=dtype, device=d)

# 해당 위치에 대한 Jacobian 계산
J = chain.jacobian(th, locations=loc)
```

Jacobian은 역운동학(IK)을 수행하는 데 사용할 수 있습니다.(참고 [IK survey](https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf))
IK는 다른 방법을 통해 더 잘 수행될 수도 있습니다. (하지만 Jacobian을 통해 수행하면 종단 간 미분 가능한 방법을 얻을 수 있습니다)

## 역운동학(IK)

역운동학(Inverse Kinematics)은 감쇠 최소자승법(damped least squares) 을 통해 구현되어 있습니다 
(야코비안의 유사역행렬에 감쇠(Jacobian pseudo-inverse damped)를 적용하여 특이점 근처에서의 진동을 방지하며 반복적으로 계산합니다).

다른 IK 라이브러리들과 비교했을 때, 본 라이브러리는 다음과 같은 장점을 갖습니다:
- ROS에 의존하지 않음 (다수의 IK 라이브러리는 로봇 설명이 ROS parameter server에 있어야 동작함)
- 목표 포즈와 초기 위치 재시도를 배치(batch)로 처리 가능
- 목표 위치(goal position) 뿐만 아니라 목표 방향(goal orientation) 도 함께 고려 가능

![IK](https://i.imgur.com/QgaUME9.gif)

사용 방법은 'tests/test_inverse_kinematics.py'를 참조하세요. 하지만 일반적으로 필요한 것은 아래와 같습니다:

```python
# URDF 파일의 전체 경로 생성
full_urdf = os.path.join(search_path, urdf)

# URDF 파일에서 직렬 체인을 구성, 엔드 이펙터 지정
chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")

# goals are specified as Transform3d poses in the **robot frame**
# 목표 위치(goal)는 **로봇 좌표계 기준**의 Transform3d 형태로 지정되어야 함
# so if you have the goals specified in the world frame, you also need the robot frame in the world frame
# 따라서 goal이 월드 좌표계로 주어졌다면, 로봇 프레임이 월드 상 어디에 있는지도 알아야 함

# 로봇 프레임이 월드 좌표계 원점에 있다고 가정 (변환 없음)
pos = torch.tensor([0.0, 0.0, 0.0], device=device)
rot = torch.tensor([0.0, 0.0, 0.0], device=device)  # 회전 없음 (Euler angles)
rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

# specify goals as Transform3d poses in world frame
# 목표 위치(goal)를 월드 좌표계에서 정의했다고 가정
goal_in_world_frame_tf = ...

# convert to robot frame (skip if you have it specified in robot frame already, or if world = robot frame)
# goal을 로봇 좌표계로 변환 (월드=로봇이면 생략 가능)
goal_in_rob_frame_tf = rob_tf.inverse().compose(goal_tf)

# get robot joint limits
# 로봇의 관절 각도 제한을 불러옴 (min/max 값들)
lim = torch.tensor(chain.get_joint_limits(), device=device)

# create the IK object
# 역운동학 객체 생성
# see the constructor for more options and their explanations, such as convergence tolerances
# 생성자에는 수렴 조건 등 다양한 설정 가능
ik = pk.PseudoInverseIK(
    chain,
    max_iterations=30,               # 최대 반복 횟수
    num_retries=10,                  # 여러 초기값에서 재시도 횟수
    joint_limits=lim.T,              # 관절 제한 (Transposed: [min, max])
    early_stopping_any_converged=True,       # 어떤 시도라도 수렴하면 조기 종료
    early_stopping_no_improvement="all",     # 전부 수렴 정지 시 중단
    debug=False,                             
    lr=0.2                           # 역운동학에서 사용할 학습률
)

# solve IK
# 역운동학 문제 해결
sol = ik.solve(goal_in_rob_frame_tf)

# num goals x num retries x DOF tensor of joint angles;
# if not converged, best solution found so far
# 목표 수 × 재시도 수 × 관절 수 크기의 텐서 반환 (미수렴 시 가장 좋은 값 반환)
print(sol.solutions)

# num goals x num retries can check for the convergence of each run
# 각 실행의 수렴 여부 확인 (True/False)
print(sol.converged)

# num goals x num retries can look at errors directly
# 위치 오차 및 회전 오차를 직접 확인 가능
print(sol.err_pos)
print(sol.err_rot)

```

## SDF Queries
최신 정보는 [pytorch-volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric)을 참조하세요. 몇 가지 instruction은 여기에 첨부되어 있습니다:

충돌 검사와 같은 많은 응용 프로그램에서는 특정 구성에서 다중 링크 로봇의 SDF를 사용하는 것이 유용합니다.

먼저, [pytorch kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics)을 사용하여 로봇 모델(URDF, SDF, MJCF 등)을 만듭니다.
예를 들어, pybullet data에서 KUKA 7 DOF 팔 모델을 사용해보겠습니다.

```python
import os
import torch
import pybullet_data
import pytorch_kinematics as pk
import pytorch_volumetric as pv

# URDF 파일 경로 설정
urdf = "kuka_iiwa/model.urdf"
search_path = pybullet_data.getDataPath()
full_urdf = os.path.join(search_path, urdf)

# URDF를 이용하여 직렬 체인 생성, 엔드 이펙터는 "lbr_iiwa_link_7"
chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")

# CUDA 사용 가능 시 GPU 사용
d = "cuda" if torch.cuda.is_available() else "cpu"

# 체인을 디바이스로 이동 (CPU 또는 GPU)
chain = chain.to(device=d)

# paths to the link meshes are specified with their relative path inside the URDF
# URDF 내부의 링크 메시(mesh) 경로는 상대 경로로 지정되어 있음

# we need to give them the path prefix as we need their absolute path to load
# 따라서 실제 메시 파일을 로드하려면 절대 경로가 필요하므로 prefix를 따로 지정해야 함

# RobotSDF 객체를 생성 (SDF 계산 및 query에 사용), 메시 경로의 절대 위치를 prefix로 지정
s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"))

```

기본적으로 각 링크에는 `MeshSDF`가 사용됩니다. 대신 더 빠른 쿼리를 위해 `CachedSDF`을 사용하려면:

```python
s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"),
                link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=1.0, device=d))
```

`y=0.02` SDF slice가 시각화될 때:

![sdf slice](https://i.imgur.com/Putw72A.png)

해당하는 surface points:

![wireframe](https://i.imgur.com/L3atG9h.png)
![solid](https://i.imgur.com/XiAks7a.png)

이 SDF에 대한 Query는 관절 구성(joint configuration) 에 따라 달라집니다 (default는 모든 관절이 0으로 설정됨).
**query는 여러 관절 구성과 query points에 대해 배치(batch)로 수행됩니다.**
예를 들어, 우리는 query하려는 여러개 관절 구성들을 배치로 가지고 있을 수 있습니다.

```python
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)
N = 200
th_perturbation = torch.randn(N - 1, 7, device=d) * 0.1
# N x 7 joint values
th = torch.cat((th.view(1, -1), th_perturbation + th))
```

또한 쿼리할 포인트 배치(각 configuration에 대해 동일한 포인트):

```python
y = 0.02
query_range = np.array([
    [-1, 0.5],
    [y, y],
    [-0.2, 0.8],
])
# M x 3 points
coords, pts = pv.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)
```

joint configurations와 query 배치를 설정합니다:

```python
s.set_joint_configuration(th)
# N x M SDF value
# N x M x 3 SDF gradient
sdf_val, sdf_grad = s(pts)
```

# 저작권 및 출처
- `pytorch_kinematics/transforms`는 [pytorch3d](https://github.com/facebookresearch/pytorch3d)에서 추출되었으며, 약간의 수정이 가해졌습니다.
  설치가 어려운 pytorch3d를 전체 의존성으로 포함시키지 않기 위해 필요한 부분만 가져왔습니다.
  로봇공학에서 일반적인 좌곱(left-multiplied) 변환(T * pt)을 사용한다는 점이 주요 차이입니다.
- `pytorch_kinematics/urdf_parser_py`, `pytorch_kinematics/mjcf_parser`는 [kinpy](https://github.com/neka-nat/kinpy)에서 추출되었으며, FK 로직도 포함합니다.
  본 저장소는 해당 로직을 PyTorch로 포팅하고 병렬화를 지원하며 몇 가지 확장을 추가했습니다.