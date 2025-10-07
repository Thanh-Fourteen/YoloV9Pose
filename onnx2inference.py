import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
    Args:
        x (np.ndarray): Input boxes in [x, y, w, h] format.
    Returns:
        np.ndarray: Converted boxes in [x1, y1, x2, y2] format.
    """
    if x.shape[-1] != 4:
        raise ValueError(f"Expected 4 columns, got {x.shape[-1]}")

    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def clip_boxes(boxes: np.ndarray, shape: tuple, step=2) -> None:
    """
    Clip bounding boxes to image shape (height, width).

    Args:
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format.
        shape (tuple): Image shape (height, width).
    """
    if boxes.shape[0] == 0:
        return

    boxes[:, 0::step] = boxes[:, 0::step].clip(0, shape[1])  # x1, x2
    boxes[:, 1::step] = boxes[:, 1::step].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, kpt_label=False, step=2):
    """
    Rescale boxes from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Shape of the image after resizing (height, width).
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2]
        img0_shape (tuple): Original image shape (height, width).
        ratio_pad (Optional[tuple]): Ratio and padding used for resizing.
            If None, it will be calculated from img0_shape.
        kpt_label (boolean): Facial landmark
        step (int): num of walking [x1, y1, v1, x2, y2, v2, ...]

    Returns:
        np.ndarray: Rescaled bounding boxes in [x1, y1, x2, y2] format.
    """
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if not kpt_label:
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, [0, 2]] /= gain
        boxes[:, [1, 3]] /= gain
        clip_boxes(boxes, img0_shape)
    else:
        boxes[:, 0::step] -= pad[0]  # x padding
        boxes[:, 1::step] -= pad[1]  # y padding
        boxes[:, 0::step] /= gain
        boxes[:, 1::step] /= gain
        clip_boxes(boxes, img0_shape, step=step)

    return boxes


def draw_predictions(
    predictions,
    orig_img,
    thickness=2,
    font_scale=0.5,
    font_thickness=1,
    kpt_label=False,
):
    for line in predictions:
        x_min, y_min, x_max, y_max, score, cls_id = line[:6]
        score = round(score, 2)
        cls_id = int(cls_id)

        x_min, y_min, x_max, y_max, cls_id = map(
            int, [x_min, y_min, x_max, y_max, cls_id]
        )
        label = f"classID-{cls_id}: {score:.2f}"
        cv2.rectangle(
            orig_img,
            (x_min, y_min),
            (x_max, y_max),
            color=colors(cls_id, True),
            thickness=thickness,
        )
        cv2.putText(
            orig_img,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colors(cls_id, True),
            font_thickness,
        )

        if kpt_label:
            kpts = line[6:]
            kpts = np.array(kpts).reshape(-1, 3)
            orig_img = draw_landmark(orig_img, kpts, 2)

    return orig_img


def draw_landmark(image, kpts, radius=3, threshold=0.5, colors=None):
    if kpts.ndim == 2 and kpts.shape[1] == 3:
        pts = kpts
    else:
        k = len(kpts) // 3 if kpts.ndim == 1 else kpts.shape[1] // 3
        pts = kpts.reshape(-1, k, 3)[0]  # (K,3)

    K = pts.shape[0]

    if colors is None:
        if colors is None:
            colors = [
                (255, 0, 0),  # red
                (0, 255, 0),  # green
                (0, 0, 255),  # blue
                (255, 255, 0),  # yellow
                (255, 0, 255),  # pink
            ]
        colors = colors[:K]
    for i, (x, y, v) in enumerate(pts):
        if v < threshold:  # visibility
            continue
        color = colors[i % len(colors)]
        cv2.circle(image, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

    return image


class ONNXIKeypoint:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape

        if isinstance(h, str) or isinstance(w, str):
            print(
                "[WARNING]: Model input shape must be static, not dynamic. Please check the ONNX model."
            )
            h, w = 640, 640
        self.model_inpsize = (w, h)

        print("Model initalize sucess with provider: ", self.model.get_providers())

    def inference(
        self, img, thresold:float=0.25, class_id_filter: Optional[List[int]] = []
    ) -> List[List[float]]:
        tensor, _, _ = self.preprocess(img, new_shape=self.model_inpsize)
        _tensor = np.expand_dims(tensor, axis=0)

        # model prediction
        outputs = self.model.run(self.opt_name, dict(zip(self.inp_name, _tensor)))
        print("\n", outputs.shape, "\n")

        # predictions = self.postprocess(
        #     preds=outputs,
        #     img=tensor,
        #     org_img=img,
        #     det_thresh=thresold,
        #     class_id_filter=class_id_filter,
        # )
        # return predictions

    def preprocess(
        self, im: np.array, new_shape=(640, 640), color=(114, 114, 114), scaleup=True
    ) -> list:
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255

        return im, r, (dw, dh)

    def postprocess(
        self,
        preds: List[np.ndarray],
        img: np.ndarray,
        org_img: np.ndarray,
        class_id_filter: Optional[List[int]],
        det_thresh: float = 0.25,
    ) -> List[List[float]]:
        """
        Postprocess the predictions (bounding boxes, scores, and classes).

        Args:
            preds (np.ndarray): The raw predictions from the model.
            img (np.ndarray): The preprocessed image.
            org_img (np.ndarray): The original image.

        Returns:
            List[List[float]]: Processed bounding boxes and scores.
        """

        bboxes, scores = preds_det[..., :4], preds_det[..., 4:]
        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = xywh_to_xyxy(bbox)
            bbox = scale_boxes(img.shape[2:], bbox, org_img.shape)
            if len(preds_kpts) > 0:
                nkpts = scale_boxes(
                    img.shape[2:],
                    preds_kpts[i],
                    org_img.shape,
                    kpt_label=True,
                    step=3,
                )

            cls, score = np.argmax(scores[i], axis=-1), np.max(scores[i], axis=-1)
            cls, score = np.expand_dims(cls, axis=-1), np.expand_dims(score, axis=-1)

            idx = score.squeeze(-1) > det_thresh

            if len(class_id_filter) > 1:
                class_id_filtered = np.isin(cls.squeeze(), class_id_filter)
                idx = np.logical_and(idx, class_id_filtered)
            if len(preds_kpts) > 0:
                pred = np.hstack([bbox, score, cls, nkpts])
            else:
                pred = np.hstack(
                    [bbox, score, cls]
                )  # Concatenate bbox, score, cls for each entry

            results.extend(pred[idx].tolist())  # Only take the valid predictions

        return results


if __name__ == "__main__":
    import os
    import glob
    from tqdm import tqdm

    output_dir = "runs/onnx"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    model = ONNXIKeypoint("weights/gelan-keypoint-ddtect/weights/best-end2end.onnx")

    samples = glob.glob("data/images/*.jpg")
    for i in tqdm(range(len(samples)), desc="Predict ONNX"):
        img = cv2.imread(samples[i])
        result = model.inference(img, thresold=0.25)

        # img = draw_predictions(result, img, kpt_label=True)
        # cv2.imwrite(os.path.join(output_dir, os.path.basename(samples[i])), img)
