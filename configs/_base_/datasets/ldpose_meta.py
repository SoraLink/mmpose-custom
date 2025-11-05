# LDPose-25 metainfo（COCO-17 + 8 个残肢端点）
dataset_info = dict(
    dataset_name='ldpose25',
    paper_info=dict(
        author='Ying et al.',
        title='LDPose: Towards Inclusive Human Pose Estimation for Limb-Deficient Individuals in the Wild',
        container='—',
        year='2025',
        homepage='—',
    ),
    keypoint_info={
        0:  dict(name='nose', id=0,  color=[51,153,255], type='upper', swap=''),
        1:  dict(name='left_eye',  id=1,  color=[51,153,255], type='upper', swap='right_eye'),
        2:  dict(name='right_eye', id=2,  color=[51,153,255], type='upper', swap='left_eye'),
        3:  dict(name='left_ear',  id=3,  color=[51,153,255], type='upper', swap='right_ear'),
        4:  dict(name='right_ear', id=4,  color=[51,153,255], type='upper', swap='left_ear'),
        5:  dict(name='left_shoulder',  id=5,  color=[0,255,0],   type='upper', swap='right_shoulder'),
        6:  dict(name='right_shoulder', id=6,  color=[255,128,0], type='upper', swap='left_shoulder'),
        7:  dict(name='left_elbow',     id=7,  color=[0,255,0],   type='upper', swap='right_elbow'),
        8:  dict(name='right_elbow',    id=8,  color=[255,128,0], type='upper', swap='left_elbow'),
        9:  dict(name='left_wrist',     id=9,  color=[0,255,0],   type='upper', swap='right_wrist'),
        10: dict(name='right_wrist',    id=10, color=[255,128,0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip',       id=11, color=[0,255,0],   type='lower', swap='right_hip'),
        12: dict(name='right_hip',      id=12, color=[255,128,0], type='lower', swap='left_hip'),
        13: dict(name='left_knee',      id=13, color=[0,255,0],   type='lower', swap='right_knee'),
        14: dict(name='right_knee',     id=14, color=[255,128,0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle',     id=15, color=[0,255,0],   type='lower', swap='right_ankle'),
        16: dict(name='right_ankle',    id=16, color=[255,128,0], type='lower', swap='left_ankle'),

        # 8 个残肢端点（互斥配对：自然关节 vs 残肢端点）
        17: dict(name='above_left_elbow_res',  id=17, color=[200,200,200], type='residual_upper', swap='above_right_elbow_res'),
        18: dict(name='above_right_elbow_res', id=18, color=[200,200,200], type='residual_upper', swap='above_left_elbow_res'),
        19: dict(name='below_left_elbow_res',  id=19, color=[200,200,200], type='residual_upper', swap='below_right_elbow_res'),
        20: dict(name='below_right_elbow_res', id=20, color=[200,200,200], type='residual_upper', swap='below_left_elbow_res'),
        21: dict(name='above_left_knee_res',   id=21, color=[180,180,180], type='residual_lower', swap='above_right_knee_res'),
        22: dict(name='above_right_knee_res',  id=22, color=[180,180,180], type='residual_lower', swap='above_left_knee_res'),
        23: dict(name='below_left_knee_res',   id=23, color=[180,180,180], type='residual_lower', swap='below_right_knee_res'),
        24: dict(name='below_right_knee_res',  id=24, color=[180,180,180], type='residual_lower', swap='below_left_knee_res'),
    },
    skeleton_info={
        # 复用 COCO 连线
        0:  dict(link=('left_ankle','left_knee'), id=0,  color=[0,255,0]),
        1:  dict(link=('left_knee','left_hip'),   id=1,  color=[0,255,0]),
        2:  dict(link=('right_ankle','right_knee'), id=2, color=[255,128,0]),
        3:  dict(link=('right_knee','right_hip'), id=3,  color=[255,128,0]),
        4:  dict(link=('left_hip','right_hip'),   id=4,  color=[51,153,255]),
        5:  dict(link=('left_shoulder','left_hip'), id=5, color=[51,153,255]),
        6:  dict(link=('right_shoulder','right_hip'), id=6, color=[51,153,255]),
        7:  dict(link=('left_shoulder','right_shoulder'), id=7, color=[51,153,255]),
        8:  dict(link=('left_shoulder','left_elbow'), id=8, color=[0,255,0]),
        9:  dict(link=('right_shoulder','right_elbow'), id=9, color=[255,128,0]),
        10: dict(link=('left_elbow','left_wrist'), id=10, color=[0,255,0]),
        11: dict(link=('right_elbow','right_wrist'), id=11, color=[255,128,0]),
        12: dict(link=('left_eye','right_eye'),   id=12, color=[51,153,255]),
        13: dict(link=('nose','left_eye'),        id=13, color=[51,153,255]),
        14: dict(link=('nose','right_eye'),       id=14, color=[51,153,255]),
        15: dict(link=('left_eye','left_ear'),    id=15, color=[51,153,255]),
        16: dict(link=('right_eye','right_ear'),  id=16, color=[51,153,255]),
        17: dict(link=('left_ear','left_shoulder'), id=17, color=[51,153,255]),
        18: dict(link=('right_ear','right_shoulder'), id=18, color=[51,153,255]),

        # 可视化残肢连线（仅作显示，不影响训练）
        19: dict(link=('left_shoulder', 'above_left_elbow_res'), id=19, color=[200, 200, 200]),
        20: dict(link=('right_shoulder', 'above_right_elbow_res'), id=20, color=[200, 200, 200]),
        21: dict(link=('left_shoulder', 'below_left_elbow_res'), id=21, color=[200, 200, 200]),
        22: dict(link=('right_shoulder', 'below_right_elbow_res'), id=22, color=[200, 200, 200]),

        23: dict(link=('left_hip', 'above_left_knee_res'), id=23, color=[180, 180, 180]),
        24: dict(link=('right_hip', 'above_right_knee_res'), id=24, color=[180, 180, 180]),
        25: dict(link=('left_hip', 'below_left_knee_res'), id=25, color=[180, 180, 180]),
        26: dict(link=('right_hip', 'below_right_knee_res'), id=26, color=[180, 180, 180]),
    },
    # 25 个关节的训练权重（先全部 1.0，或略提高残肢端点的权重也可）
    joint_weights=[1.0]*25,

    # OKS sigma：前 17 复用 COCO；残肢 8 点给近似值（肘/腕/膝/踝的相近尺度）
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        # 残肢端点（近似）：above_elbow≈elbow, below_elbow≈wrist,
        # above_knee≈knee, below_knee≈ankle
        0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
    ]
)