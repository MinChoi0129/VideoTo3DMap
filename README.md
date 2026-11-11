# VideoTo3DMap

영상 하나를 넣으면 3D 지도(점 구름과 keyframe 카메라 자세)가 나오는 라이브러리다. 지도의 형상은 VGGT-Omega가 만든다. keyframe을 64장씩, 이웃과 32장이 겹치게 묶은 window마다 depth와 카메라를 한 번에 추론하고, 이웃한 window는 겹친 keyframe의 모든 화소로 맞춘 Sim(3)로 잇는다.

VGGT-SLAM 2.0에서는 다시 지나간 곳을 묶는 loop closure를 가져왔다. 비슷한 keyframe 쌍을 SALAD 검색으로 고르고 VGGT-SLAM 2.0에 들어 있는 VGGT(1B)의 attention match ratio로 같은 장소인지 검증한 뒤, 받아들인 쌍을 VGGT-Omega로 다시 재구성해 두 window 사이의 제약을 만들고 모든 window의 Sim(3)를 같은 가중의 graph로 한 번에 푼다. VGGT는 이 검증 단계에만 쓰며 지도에는 들어가지 않는다.
