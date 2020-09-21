import copy
from collections import namedtuple
from typing import Type

import cv2
import numpy as np
from zdl.AI.pose.pose.pose import Poses, Pose
from zdl.utils.io.log import logger
from zdl.utils.media.image import ImageCV
from zdl.utils.media.point import Point


class DatumPickleable:
    def __init__(self, datum, model_type: Type[Pose], add_inherit_flag_col: bool = False, title=''):
        self.poseKeypoints = self.addInheritFlagCol(datum.poseKeypoints) \
            if add_inherit_flag_col else datum.poseKeypoints
        self.cvInputData = datum.cvInputData
        self.cvOutputData = datum.cvOutputData
        self.model_type = model_type
        self.title = title
        self.text = ''
        self.poseKeypoints_stable = None

    def __getstate__(self):
        state = {
            'poseKeypoints': self.poseKeypoints,
            'cvInputData': self.cvInputData,
            'cvOutputData': self.cvOutputData,
        }
        return state

    def __setstate__(self, state):
        self.poseKeypoints = state['poseKeypoints']
        self.cvInputData = state['cvInputData']
        self.cvOutputData = state['cvOutputData']

    def showState(self):
        logger.debug(self.poseKeypoints)
        ImageCV(self.cvInputData, 'cvInputData').show()
        ImageCV(self.cvOutputData, 'cvOutputData').show()

    @classmethod
    def addInheritFlagCol(cls, poseKeypoints):
        shape = poseKeypoints.shape
        if shape == () or shape and shape[-1] == 4: return poseKeypoints
        append_shape = *shape[:-1], 1
        result = np.concatenate((poseKeypoints, np.zeros(append_shape, dtype=poseKeypoints.dtype)), axis=2)
        return result

    def putText(self, text, point=(50, 50), font_scale=None, thickness=None, img=None):
        text = str(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if img is None:
            img = self.cvOutputData
        if img.shape[0] > 800:
            font_scale = font_scale or 2
            thickness = thickness or 3
        elif img.shape[0] > 400:
            font_scale = font_scale or 1.4
            thickness = thickness or 2
        else:
            font_scale = font_scale or 0.4
            thickness = thickness or 1
        point = type(point)(map(int, point))
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        left_bottom = (point[0] - thickness * 2, point[1] + thickness * 2)
        right_up = (point[0] + text_width + thickness * 2, point[1] - text_height - thickness * 2)
        cv2.rectangle(img, left_bottom, right_up, (255, 255, 255), -1)
        cv2.putText(img, text, point, font, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA, False)

    def appendText(self, text, point=(50, 50)):
        self.text += f'\n{text}'
        cv2.putText(self.cvOutputData, self.text, point, cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (102, 0, 255), 2, cv2.LINE_AA, False)

    def centers(self, need_type=list):
        clean_pose = self.poseKeypoints.copy()
        Poses(clean_pose, pose_type=self.model_type).cleanup(['face', 'feet'])
        return Point.pointsCenter(clean_pose, need_type=need_type)

    def countNonzeroPoints(self):
        return [np.count_nonzero(np.count_nonzero(pose[:, :2], axis=1)) for pose in self.poseKeypoints]

    def poseKeypointsOrderByCenterX(self):
        return self.poseKeypoints[np.argsort(self.centers())]

    def poseKeypointsOrderByIntegrity(self):
        return sorted(self.poseKeypoints, key=lambda x: np.count_nonzero(x[:, 0]), reverse=True)

    def labelPoints(self, points, img=None, mark=False, font_scale=None, thickness=None, copy_=True):
        if img is None:
            img = self.cvOutputData
        if copy_:
            img = copy.deepcopy(img)
        if mark:
            for i, p in enumerate(points):
                self.putText(i, tuple(p[:2]), font_scale=font_scale, thickness=thickness, img=img)
        else:
            for p in points:
                cv2.circle(img, (p[0], p[1]), 15, (0, 255, 255), thickness=2, lineType=8, shift=0)
        self.show(img)

    def show(self, img, title=None):
        ImageCV(img, title).show()

    def showCvOut(self):
        ImageCV(self.cvOutputData, self.title).show()

    @classmethod
    def rebuildFromRoiDatum(cls, img, roi_datum_tuple_list, model_type, add_inherit_flag_col: bool = False):
        datum_rebuild = namedtuple('datum_rebuild', ['poseKeypoints', 'cvInputData', 'cvOutputData'])
        datum_rebuild.cvInputData = img
        img_fill = np.copy(img)
        datums_pk = []
        for roi_rect, datum in roi_datum_tuple_list:
            if datum.poseKeypoints.shape == ():
                add_one_zero_pose = np.zeros((1, 25, 4), dtype=np.float32)
                datums_pk.append(add_one_zero_pose)
                continue

            ymin, xmin, ymax, xmax = roi_rect
            img_fill[ymin:ymax, xmin:xmax] = datum.cvOutputData
            pks_incr = np.zeros(datum.poseKeypoints.shape, dtype=np.float32)
            pks_incr[..., 0] = xmin
            pks_incr[..., 1] = ymin
            pks_incr[datum.poseKeypoints == 0] = 0
            datums_pk.append(datum.poseKeypoints + pks_incr)

        datum_rebuild.poseKeypoints = np.vstack(datums_pk) if datums_pk else np.array(0.0)
        datum_rebuild.cvOutputData = img_fill
        return cls(datum_rebuild, model_type, add_inherit_flag_col)
