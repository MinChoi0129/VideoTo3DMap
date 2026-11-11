"""Keyframe selection on the raw video and the window layout over the keyframes."""
import cv2
import numpy as np


class SummedFlowKeyframes:
    """A frame becomes a keyframe once the image motion summed frame by frame since the last keyframe exceeds min_disparity px.
    Each step tracks freshly detected corners from the previous frame (Lucas-Kanade), so fast turns are not hidden by points that
    leave the view, which is what makes keyframes jump metres apart when tracking from the last keyframe only."""

    def __init__(self, min_disparity):
        self.min_disparity, self.prev, self.acc = min_disparity, None, 0.0

    def __call__(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p0 = None if self.prev is None else cv2.goodFeaturesToTrack(self.prev, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)
        new = p0 is None or len(p0) < 10
        if not new:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(21, 21), maxLevel=3); ok = st.ravel() == 1; new = ok.sum() < 10
            if not new:
                self.acc += float(np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1).mean()); new = self.acc > self.min_disparity
        self.prev = g; self.acc = 0.0 if new else self.acc
        return new


def read_keyframes(video, min_disparity, stride=1):
    """Returns (frame indices, BGR images) of the keyframes of a video file."""
    cap, pick, ids, imgs, i = cv2.VideoCapture(video), SummedFlowKeyframes(min_disparity), [], [], 0
    if not cap.isOpened():
        raise FileNotFoundError(video)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0 and pick(img):
            ids.append(i); imgs.append(img)
        i += 1
    cap.release()
    return np.array(ids), imgs


def window_starts(n, size, overlap):
    """First keyframe of every window; the last window is moved back so that it ends on the last keyframe."""
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, size - overlap))
    return starts + ([n - size] if starts[-1] + size < n else [])
