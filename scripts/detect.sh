# python detect.py \
#     --weights 'runs/train/gelan-keypoint2/weights/best.pt' \
#     --source 'data/images/selfie.jpg' \
#     --conf-thres 0.1 \
#     --iou-thres 0.45 \
#     --device '0' \
#     --kpt-label 5

python detect_kpts.py \
    --weights 'runs/train/gelan-keypoint-ddtect/weights/best.pt' \
    --source 'data/images/selfie.jpg' \
    --img 640 \
    --device '0' \
    --kpt-label 5 \
    --hide-labels