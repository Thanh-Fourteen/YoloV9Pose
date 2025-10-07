# python train_dual.py \
#     --weights 'weights/yolov9-c.pt' \
#     --cfg 'models/detect/yolov9-c.yaml' \
#     --data 'datahub/face-gender-dataset-v2/data.yaml' \
#     --batch-size 8 \
#     --name "yolov9-c"


# python train.py \
#     --weights 'weights/gelan-c.pt' \
#     --cfg 'models/detect/gelan-c.yaml' \
#     --data 'datahub/face-gender-dataset-v2/data.yaml' \
#     --hyp 'data/hyps/hyp.scratch-high.yaml' \
#     --batch-size 8 \
#     --name "gelan"

python train.py \
    --weights 'weights/gelan-c.pt' \
    --cfg 'models/detect/gelan-keypoint-ddtect.yaml' \
    --data '/home/tiennv/tiennv/FPT/thanhnn/datasets/face_gender/data.yaml' \
    --hyp 'data/hyps/hyp.scratch-high.yaml' \
    --batch-size 8 \
    --name "gelan-keypoint-ddtect" \
    --kpt-label 5 \
    --epochs 250 \
    --optimizer 'AdamW' \

    # --data 'datahub/widerface.yaml' \