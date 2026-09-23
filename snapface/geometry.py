"""Géométrie : conversions cv2<->torch et math des crops."""


def to_tensor(bgr, device):
    raise NotImplementedError


def from_tensor(t):
    raise NotImplementedError


def crop_face(frame, det, expand=0.35):
    raise NotImplementedError


def paste_face(frame, face_img, box, feather=15):
    raise NotImplementedError


def soft_mask(h, w, feather_frac=0.35, device="cpu"):
    raise NotImplementedError


def clamp_bbox(box, frame_shape):
    raise NotImplementedError