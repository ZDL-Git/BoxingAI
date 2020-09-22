import inspect
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from multiprocessing import Process, Queue
from typing import Optional, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import psutil
# pylab.rcParams['font.sans-serif']=['SimHei'] # 为了支持中文，但是需要重启kernel，未采用
# pylab.rcParams['axes.unicode_minus']=False
# pylab.rcParams['animation.embed_limit']=50
import pylab
from IPython.core.display import HTML, display
from matplotlib import animation
from zdl.AI.helper.openpose import DatumPickleable
from zdl.AI.object_detection.TF_detector import ObjectDetector
from zdl.AI.pose.extractor.extractor import Extractor
from zdl.AI.pose.pose.pose import Poses
from zdl.utils.helper.numpy import ndarrayLen
from zdl.utils.io.log import logger
from zdl.utils.media.image import ImageCV
from zdl.utils.media.point import Point
from zdl.utils.media.video import Video
from zdl.utils.time.counter import timeit

from zdl.projects.BoxingAI.plotting_pickleable import PlottingPickleable


class BoxingAIHelper:
    def __init__(self):
        self._boxer_detector = None
        self._pose_estimator = None
        self.queue_name_img_tuple = None
        self.media = None
        self.prep_hooks = []
        self.show = False

    def _sortBoxerByCenterX(self, posescore_list):
        sorted_ = sorted(posescore_list, key=lambda t: t.pose.center()[0])
        return sorted_

    def _isPointInsideRect(self, point, rect):
        return rect[1] <= point[0] <= rect[3] and rect[0] <= point[1] <= rect[2]

    def _refineBoxer(self, boxer_entities):
        # merge covered boxer
        for i, boxer_entity_i in enumerate(boxer_entities):
            for j, boxer_entity_j in enumerate(boxer_entities[i + 1:]):
                pass
        return boxer_entities


class BoxingAI(BoxingAIHelper, metaclass=ABCMeta):
    PoseScore = namedtuple('PoseScore', ['pose', 'boxer_id', 'completion_score', 'completion_multi_boxerScore',
                                         'points_score_sum', 'points_scores_sum_after_re_pu',
                                         'norm_dis_to_boxer_center',
                                         'knee_and_below_nonzero_exists'])

    def __init__(self):
        super().__init__()
        self.indices = None

    @abstractmethod
    def _startProducingImgs(self):
        pass

    def setBoxerDetector(self, object_detector: ObjectDetector):
        self._boxer_detector = object_detector
        self._boxer_detector.modelInfo()
        return self

    def setPoseEstimator(self, pose_estimator: Extractor):
        self._pose_estimator = pose_estimator
        return self

    def setShow(self, show: bool):
        if show and self.__class__ == BoxingAIVideo:
            num_to_show = len(self.indices) if self.indices else self.media.get_info()['frame_c']
            if num_to_show > 10 and input(f"About {num_to_show} pictures to be displayed, which may lead to crash."
                                          " Do you want to continue? [y/n]:") != 'y':
                sys.exit('exit.')
        self.show = show
        return self

    def addPrepHooks(self, hooks: list):
        # hook should be a function, receive a img param then return_ the processed img
        self.prep_hooks += hooks
        return self

    def _execPrepHooks(self, img, name):
        for i, hook in enumerate(self.prep_hooks):
            logger.debug(f'hook {i} running...\n{inspect.getsource(hook)}')
            img = hook(img)
            self._showIfEnabled(img, title=f'{name}:img_load_hook {i} result')
        return img

    @timeit
    def _detectBoxer(self, name_img_tuple):
        result, box_entities = self._boxer_detector.detect(name_img_tuple[1])
        imgobj = ImageCV(name_img_tuple[1])
        for box_entity in box_entities:
            box_entity[0][::] = imgobj.normRectToAbsRect(box_entity[0], 1)

        if self.show:
            imgobj.setTitle(f'{name_img_tuple[0]}:boxer_detect_result:').drawBoxes(box_entities, copy=True).show()
        return result, box_entities

    def _processImg(self, image) -> Tuple:
        poses, datum = self._pose_estimator.extract(image)
        return poses, datum

    @timeit
    def _estimatePose(self, name_img_tuple):
        poses, datum = self._processImg(name_img_tuple[1])

        if self.show:
            center_points = Poses(datum.poseKeypoints, self._pose_estimator.pose_type).centers(
                need_type=tuple)
            ImageCV(datum.cvOutputData).drawPoints(center_points, copy=False)
            self._showIfEnabled(datum.cvOutputData, title=f'{name_img_tuple[0]}:pose_estimate_result')
        return datum

    @timeit
    def _gen_main_process(self, max_people_num, heuristic, smooth):
        # img_shape = self.media.get_info()['shape']
        # act_recg1 = ActionRecognizer(self.pose_estimator_env[0]['model_pose'],img_shape)
        # act_recg2 = ActionRecognizer(self.pose_estimator_env[0]['model_pose'],img_shape)
        while True:
            got = self.queue_name_img_tuple.get()
            if got == 'DONE': break
            if heuristic:
                logger.debug('heuristic mode')
                _, boxer_entities = self._detectBoxer(got)
                img_obj = ImageCV(got[1], title=got[0])
                dilate_ratio = 1.1
                roi_datum_tuple_list = []
                boxer_id_score_to_every_pose = []
                for i, b in enumerate(boxer_entities):
                    roi_rect = img_obj.rectDilate(b[0], dilate_ratio)
                    boxer_roi_img = img_obj.roiCopy(roi_rect).org()
                    datum = self._estimatePose((f'{got[0]}:boxer {i} (dilate_ratio {dilate_ratio})', boxer_roi_img))
                    roi_datum_tuple_list.append((roi_rect, datum))
                    boxer_id_score_to_every_pose += [(i, b)] * max(1, ndarrayLen(datum.poseKeypoints))
                datum = DatumPickleable.rebuildFromRoiDatum(got[1], roi_datum_tuple_list,
                                                            self._pose_estimator.pose_type, True)
                # datum,fill_num = self._fill_missing_pose_keypoints(datum, max_people_num)
                # boxer_id_score_to_every_pose += [None] * fill_num
                self._showIfEnabled(datum.cvOutputData, title=f'{got[0]}:rebuild(may be covered)')
                poses = Poses(datum.poseKeypoints, self._pose_estimator.pose_type)
                poses.cleanup(['face'])
                posescore_list = self._rescorePoseByBoxerScore(poses, boxer_id_score_to_every_pose)
                posescore_list = self._filter_out_duplicate_pose(posescore_list)
            else:
                # if heuristic and not boxer_entities:
                #     logger.warn('only one boxer detected, use full img pose estimation!')
                datum = self._estimatePose(got)
                # datum,fill_num = self._fill_missing_pose_keypoints(datum, max_people_num)
                poses = Poses(datum.poseKeypoints, self._pose_estimator.pose_type)
                poses.cleanup(['face'])
                posescore_list = self._rescorePoseByBoxerScore(poses, None)
                boxer_entities = None

            # posescore_list = self._fill_missing_pose(posescore_list, max_people_num)
            logger.debug('rescored pose score:')
            logger.debug([f'frame index:[{got[0]}] from_boxer:{t[1]}  '
                          f'completion_score(cleaned):{t[2]} completion_multi_boxerscore:{t[3]}  '
                          f'points_scores_sum:{t[4]} scores_sum_after_re_pu:{t[5]}  '
                          f'norm_dis_to_boxer_center:{t[6]} knee_and_below_nonzero_exists:{t[7]}  '
                          f'[p0x:{t[0].key_points[0][0]} p1x:{t[0].key_points[1][0]} p8x:{t[0].key_points[8][0]} p17x:{t[0].key_points[17][0]}]'
                          for t in posescore_list])
            # posescore_list = list(filter(lambda t: t.points_scores_sum_after_re_pu,posescore_list[:max_people_num]))
            # posescore_list = posescore_list[:3]
            posescore_list = self._connectAndSmoothPoses(posescore_list, smooth=smooth)

            # actions_hist1 = act_recg1.put_pose(posescore_list[0].pose)[-5:]
            # actions_hist2 = act_recg2.put_pose(posescore_list[1].pose)[-5:]
            # act_recg1.show()

            # yield got[0],list(posescore_list),datum,boxer_entities if heuristic else None,[actions_hist1,actions_hist2]
            yield got[0], list(posescore_list), datum, boxer_entities if heuristic else None, [[], []]

    def _fill_missing_pose(self, posescore_list, target_num):
        # exists_num = len(posescore_list)
        # fill_num = missing_num = target_num - exists_num
        exists_boxer_num = len({p_s.boxer_id for p_s in posescore_list})
        missing_num = target_num - exists_boxer_num
        if missing_num <= 0: return posescore_list
        logger.info(f'fill missing pose keypoints num: {missing_num}')
        for _ in range(missing_num):
            new_pose = np.zeros((25, 4), dtype=np.float32)
            posescore_list.append(self.PoseScore(new_pose, None, 0, 0, 0, 0, 1, False))
        return posescore_list

    @abstractmethod
    def _connectAndSmoothPoses(self, all_poses, smooth):
        pass

    @timeit
    def _filter_out_duplicate_pose(self, pose_score_entities):
        # poses_entities param should be a entities list, and every element's first position should be a pose
        true_means_keep = [True] * len(pose_score_entities)
        for i, pose_score_entity_i in enumerate(pose_score_entities):
            p_i, p_i_score = pose_score_entity_i.pose, pose_score_entity_i.points_scores_sum_after_re_pu
            p_i_x = p_i.key_points[..., 0]
            p_i_x_b = p_i_x > 0
            for j, pose_score_entity_j in enumerate(pose_score_entities[i + 1:]):
                p_j, p_j_score = pose_score_entity_j.pose, pose_score_entity_j.points_scores_sum_after_re_pu
                p_j_x = p_j.key_points[..., 0]
                p_j_x_b = p_j_x > 0
                p_i_j_x_b = np.logical_and(p_i_x_b, p_j_x_b)
                if_exceed_this_then_remove = min(p_i_x_b.sum(), p_j_x_b.sum()) * 0.6
                points_dis_thre = 0.05 * (p_i.torso_height() if p_i_score > p_i_score else p_j.torso_height())
                if (np.abs((p_i_x - p_j_x)[p_i_j_x_b]) < points_dis_thre).sum() > if_exceed_this_then_remove:
                    index_to_remove = i + 1 + j if p_i_score >= p_j_score else i
                    true_means_keep[index_to_remove] = False
                    logger.debug(f'Removing duplicate pose: {pose_score_entities[index_to_remove]}')
        return [pose_score_entities[i] for i in range(len(true_means_keep)) if true_means_keep[i]]

    def _rescorePoseByBoxerScore(self, poses: Poses, boxer_id_score_to_every_pose: Optional[List[Tuple]]):
        if not poses.all_keypoints.any(): return []

        reward_and_punishment = np.ones(len(poses[0].key_points))
        reward_and_punishment[poses.PARTS_INDICES['face']] = 0
        # reward_and_punishment[points_indices['knee_and_below']] = 1.2
        posescore_list = []
        for i, pose in enumerate(poses):
            points_scores = pose.key_points[..., 2]
            nonzero_scores_bool = points_scores > 0
            count_nonzero = nonzero_scores_bool.sum()
            knee_and_below_nonzero_exists = nonzero_scores_bool[poses.PARTS_INDICES['knee_and_below']].sum() > 0
            points_scores_sum = points_scores.sum()
            points_scores_sum_after_re_pu = (points_scores * reward_and_punishment).sum()
            if boxer_id_score_to_every_pose and boxer_id_score_to_every_pose[i]:
                boxer_id, boxer_entity = boxer_id_score_to_every_pose[i]
                boxer_rect, boxer_score = boxer_entity[0], boxer_entity[2]
                diagonal_half = ((boxer_rect[3] - boxer_rect[1]) ** 2 + (boxer_rect[2] - boxer_rect[0]) ** 2) ** 0.5 / 2
                boxer_center = Point((boxer_rect[1] + boxer_rect[3]) / 2, (boxer_rect[0] + boxer_rect[2]) / 2)
                pose_center = pose.center()
                norm_dis_to_boxer_center = Point.dis(boxer_center, pose_center) / diagonal_half
            else:
                boxer_id, boxer_score = None, 1
                # from heuristic denotes punishment, from non heuristic will be idle
                norm_dis_to_boxer_center = 1
            pose_score = self.PoseScore(pose, boxer_id, count_nonzero, count_nonzero * boxer_score,
                                        points_scores_sum, points_scores_sum_after_re_pu,
                                        norm_dis_to_boxer_center,
                                        knee_and_below_nonzero_exists)
            posescore_list.append(pose_score)
        posescore_list.sort(key=lambda t: t.norm_dis_to_boxer_center, reverse=False)
        # posescore_list.sort(key=lambda t: t.points_scores_sum_after_re_pu, reverse=True)
        # posescore_list.sort(key=lambda t: t.knee_and_below_nonzero_exists, reverse=True)
        return posescore_list

    def _showIfEnabled(self, img, title=None):
        if self.show: ImageCV(img, title).show()

    @timeit
    def detectBoxer(self):
        assert self._boxer_detector, 'boxer_detector is None'
        self._startProducingImgs()
        res = []
        while True:
            got = self.queue_name_img_tuple.get()
            if got == 'DONE': break
            res.append(self._detectBoxer(got))
        logger.debug(res)
        return res

    @timeit
    def estimatePose(self):
        assert self._pose_estimator, '_pose_estimator is None'
        self._startProducingImgs()
        res = []
        while True:
            got = self.queue_name_img_tuple.get()
            if got == 'DONE': break
            res.append(self._estimatePose(got))
        logger.debug(res)
        return res

    @timeit
    def detect_boxer_then_pose_bak(self, max_people_num=2, heuristic=True, smooth=0.8):
        assert self._boxer_detector, '_boxer_detector is None'
        assert self._pose_estimator, '_pose_estimator is None'
        self._startProducingImgs()
        all_poses = []
        try:
            while True:
                cur_index, posescore_list, datum, boxer_entities, actions_hist_list = next(
                    self._gen_main_process(max_people_num, heuristic, smooth))
                all_poses.append(posescore_list)
        except StopIteration:
            pass
        return all_poses
        # return boxer_entities,datum,poses

    # @timeit
    # def evaluate(self,max_people_num=2,heuristic=True,smooth=0.8):
    #     assert self.boxer_detector, 'boxer_detector is None'
    #     assert self.pose_estimator_env, 'pose_estimator_env is None'

    #     self._start_producing_imgs()
    #     gen = self._gen_main_process(number_people_max, heuristic, smooth)
    #     all_for_debug = []
    #     try:
    #         while True:
    #             cur_index, posescore_list, datum, boxer_entities, actions_hist_list = next(gen)
    #             frame4debug = {}
    #             all_for_debug.append((posescore_list, boxer_entities))

    #     except StopIteration:
    #         pass

    #     return all_for_debug

    def extractPose(self, max_people_num=2, heuristic=True, smooth=0.8):
        assert self._boxer_detector, '_boxer_detector is None'
        assert self._pose_estimator, '_pose_estimator is None'

        self._startProducingImgs()
        gen = self._gen_main_process(max_people_num, heuristic, smooth)
        all_for_debug = []
        try:
            while True:
                cur_index, posescore_list, datum, boxer_entities, actions_hist_list = next(gen)
                all_for_debug.append((cur_index, posescore_list, boxer_entities))
        except StopIteration:
            pass
        return all_for_debug

    @timeit
    def _plotPose(self, axes, poses, points=None, texts: Optional[dict] = None, rects: list = None):
        pose_sections = self._pose_estimator.pose_type.sections
        pose_colors = ['blue', 'red', 'gray']

        line_objs = []
        if poses:
            for p_i, pose in enumerate(poses):
                pose_nonzero_b = pose.key_points != 0
                pose_x_or_y_nonzero_b = np.logical_or(pose_nonzero_b[..., 0], pose_nonzero_b[..., 1])
                for i, s in enumerate(pose_sections):
                    s_nonzero = np.asarray(s)[pose_x_or_y_nonzero_b[s]]
                    x_a = pose.key_points[s_nonzero][..., 0]
                    y_a = pose.key_points[s_nonzero][..., 1]
                    # section = ([person_trans[0][j] for j in s if person_trans[0][j]!=0],
                    #         [person_trans[1][j] for j in s if person_trans[1][j]!=0])
                    markersize = 6 if i in [0, 6, 7] else 8
                    lobj, = axes['main'].plot(x_a, y_a, ls='-', color=pose_colors[p_i], marker='o',
                                              markersize=markersize, markerfacecolor='white')
                    line_objs.append(lobj)

                # set marker of inherited points to yellow
                # logger.debug(pose[3])
                partially_inherited_points = pose.key_points[..., 3] == 1
                should_plot = np.logical_and(partially_inherited_points, pose_x_or_y_nonzero_b)
                lobj, = axes['main'].plot(pose.key_points[..., 0][should_plot], pose.key_points[..., 1][should_plot],
                                          ls='', color=pose_colors[p_i], marker='o', markersize=6,
                                          markerfacecolor=pose_colors[p_i])
                line_objs.append(lobj)
                fully_inherited_points = pose.key_points[..., 3] == 2
                should_plot = np.logical_and(fully_inherited_points, pose_x_or_y_nonzero_b)
                lobj, = axes['main'].plot(pose.key_points[..., 0][should_plot], pose.key_points[..., 1][should_plot],
                                          ls='', color=pose_colors[p_i], marker='+', markersize=10,
                                          markerfacecolor=pose_colors[p_i])
                line_objs.append(lobj)

        point_objs = []
        if points:
            if isinstance(points, list):
                points = np.asarray(points)
            # logger.debug(points)
            # plot points by line
            lobj, = axes['main'].plot(points[:, 0], points[:, 1], ls='', color='white', marker='o', markersize=10,
                                      markerfacecolor='black')
            point_objs.append(lobj)

        rect_objs = []
        if rects:
            for i, r in enumerate(rects):
                rect = plt.Rectangle((r[1], r[0]),
                                     r[3] - r[1],
                                     r[2] - r[0], fill=False,
                                     edgecolor='#e1f2e9', linewidth=2.5)
                rect_objs.append(axes['main'].add_patch(rect))

        text_objs = []
        if texts.get('main'):
            text_objs.append(axes['tip'].text(15, 15, ' '.join(texts['main']), verticalalignment='top',
                                              fontsize=18, color='black'))
        ax_tip_half_w = axes['tip'].get_xlim()[1] / 2
        boxer1_left_margin, boxer2_left_margin, boxer_top_margin = 20, 20 + ax_tip_half_w, 70
        if texts.get('boxer1'):
            text_objs.append(axes['tip'].text(boxer1_left_margin, boxer_top_margin, '\n'.join(texts['boxer1']),
                                              verticalalignment='top', fontsize=12, color=pose_colors[0]))
        if texts.get('boxer2'):
            text_objs.append(axes['tip'].text(boxer2_left_margin, boxer_top_margin, '\n'.join(texts['boxer2']),
                                              verticalalignment='top', fontsize=12, color=pose_colors[1]))

        return line_objs + point_objs + rect_objs + text_objs

    @timeit
    def animation(self, fps=23, number_people_max=2, display_=True,
                  plot_video=None, plot_center=False, plot_rect=False,
                  save_video_to=None, heuristic=True, show_animation_video=False, smooth=0.8):
        media_info = self.media.get_info()
        w, h = media_info['width'], media_info['height']
        tip_h = 300

        fig = pylab.figure()
        fig.set_figwidth(9)  # 12.8
        fig.set_figheight(9 * (h + tip_h) / w)

        left_margin, bottom_margin, width = 0.05, 0.02, 0.9
        ax_main = pylab.axes([left_margin, bottom_margin + 0.9 * tip_h / (tip_h + h), width, 0.9 * h / (tip_h + h)])
        ax_main.xaxis.set_ticks_position('top')
        ax_main.set_xlim((0, w))
        ax_main.set_ylim((h, 0))
        ax_main.xaxis.set_major_locator(pylab.MultipleLocator(100))
        ax_main.xaxis.set_minor_locator(pylab.MultipleLocator(20))
        ax_main.yaxis.set_minor_locator(pylab.MultipleLocator(20))

        ax_tip = pylab.axes([left_margin, bottom_margin, width, 0.9 * tip_h / (tip_h + h)])
        ax_tip.set_xlim((0, w))
        ax_tip.set_ylim((tip_h, 0))
        ax_tip.set_xticks([])
        ax_tip.set_yticks([])
        ax_tip.set_facecolor('#ebe6eb')

        pylab.close()

        self._startProducingImgs()
        gen = self._gen_main_process(number_people_max, heuristic, smooth)
        all_for_debug = []
        frames = []
        try:
            while True:
                cur_index, posescore_list, datum, boxer_entities, actions_hist_list = next(gen)
                all_for_debug.append((posescore_list, boxer_entities))
                frame_artists = []
                poses = [t[0] for t in posescore_list]
                texts = {'main': [f'index:{cur_index}'],
                         'boxer1': [f'> {ah.name:<12} {ah.start_index:>5}-{ah.end_index:<5} {ah.score:5.2f}' for ah in
                                    actions_hist_list[0]],
                         'boxer2': [f'> {ah.name:<12} {ah.start_index:>5}-{ah.end_index:<5} {ah.score:5.2f}' for ah in
                                    actions_hist_list[1]]}
                if plot_video:
                    im = ax_main.imshow(datum.cvOutputData[..., ::-1], animated=True)
                    frame_artists.append(im)
                points = None
                if plot_center:
                    points = [pose.center() for pose in poses]
                rects = None
                if plot_rect:
                    rects = [boxer_entity[0] for boxer_entity in boxer_entities]
                frame_artists += self._plotPose(axes={'main': ax_main, 'tip': ax_tip}, poses=poses, points=points,
                                                texts=texts, rects=rects)
                frames.append(frame_artists)
                # fig.canvas.draw()
                # fig.savefig('sssss.png')
        except StopIteration:
            pass

        anim = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True, repeat_delay=1000)

        if display_:
            # 1.控件
            # Note: pylab.rc is the part which makes it work on Colab
            pylab.rc('animation', html='jshtml')
            display(anim)
        if show_animation_video:
            pylab.rc('animation', html='jshtml')
            # 2.转换成视频
            display(HTML(anim.to_html5_video()))
        if save_video_to is not None:
            anim.save(save_video_to)
        return anim, all_for_debug

    @timeit
    def evaluate(self, number_people_max=2, heuristic=True, smooth=0.8):
        plotting = PlottingPickleable(self.media.get_info(), self._pose_estimator.pose_type.NAME)

        self._startProducingImgs()
        gen = self._gen_main_process(number_people_max, heuristic, smooth)
        all_for_debug = []
        try:
            while True:
                cur_index, posescore_list, datum, boxer_entities, actions_hist_list = next(gen)
                all_for_debug.append((posescore_list, boxer_entities))

                poses = [t[0] for t in posescore_list]
                texts = {'main': [f'index:{cur_index}'],
                         'boxer1': [f'> {ah.name:<12} {ah.start_index:>5}-{ah.end_index:<5} {ah.score:5.2f}' for ah in
                                    actions_hist_list[0]],
                         'boxer2': [f'> {ah.name:<12} {ah.start_index:>5}-{ah.end_index:<5} {ah.score:5.2f}' for ah in
                                    actions_hist_list[1]]}
                # if plot_video:
                #     im = ax_main.imshow(datum.cvOutputData[..., ::-1], animated=True)
                # frame_artists.append(im)
                points = [pose.center() for pose in poses]
                rects = [boxer_entity[0] for boxer_entity in boxer_entities]

                plotting.newFrame(poses=poses, points=points, texts=texts, rects=rects)
        except StopIteration:
            pass

        plotting.dump('plotting.pickle')

        return plotting, all_for_debug


class BoxingAIVideo(BoxingAI):
    def __init__(self, path, indices=None):
        assert os.path.isfile(path)
        super().__init__()
        self.media = Video(path)
        self.indices = indices
        self.queue_name_img_tuple = Queue(maxsize=10)
        # self.imgs_producer = partial(
        #     self._new_process_to_seed_video_frames, path, indices)
        self.prev_frame_poses_holder = None
        self.frame_poses_inherited_times = None
        self.frame_poses_fully_inherited_times = None

    def setIndices(self, indices: Union[range, Tuple, List]):
        self.indices = indices
        return self

    def _startProducingImgs(self):
        # _newProcessToSeedVideoFrames:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        logger.debug(f'parent pid: {current_process.pid}')
        logger.debug(f'children pids: {[c.pid for c in children]}')

        def _load_video(path, indices, queue):
            gen_name_img = Video(path).info().readDict(indices=indices, yield_=True)
            try:
                while True:
                    if queue.full():
                        time.sleep(1)
                    else:
                        name, img = next(gen_name_img)
                        # self.require_show(img,title=f'{name}:org')
                        img = self._execPrepHooks(img, name)
                        queue.put((name, img))
            except StopIteration:
                queue.put('DONE')

        pw = Process(target=_load_video, args=(self.media.fname, self.indices, self.queue_name_img_tuple),
                     name='imgs_producer_process')
        pw.daemon = True
        pw.start()

    def _getEachBoxerTop(self, frame_posescore_entities, fill_to_least):
        top_pose_per_boxer = {}
        frame_posescore_entities_sorted = sorted(frame_posescore_entities, key=lambda t: t.norm_dis_to_boxer_center)
        for p_s in frame_posescore_entities_sorted:
            if p_s.boxer_id not in top_pose_per_boxer:
                top_pose_per_boxer[p_s.boxer_id] = p_s
        need_fill = max(fill_to_least - len(top_pose_per_boxer), 0)
        for i in range(need_fill):
            boxer_id = 10000 + i
            new_pose = self._pose_estimator.pose_type(np.zeros((25, 4), dtype=np.float32))
            top_pose_per_boxer[boxer_id] = self.PoseScore(new_pose, boxer_id, 0, 0, 0, 0, 1, False)

        return top_pose_per_boxer

    def _getCandidates(self, frame_posescore_entities, fill_to_least):
        boxer_id_set = {p.boxer_id for p in frame_posescore_entities}
        boxer_num = len(boxer_id_set)
        if boxer_num >= 2:
            # get each boxer top
            top_pose_per_boxer = {}
            frame_posescore_entities_sorted = sorted(frame_posescore_entities, key=lambda t: t.norm_dis_to_boxer_center)
            for p_s in frame_posescore_entities_sorted:
                if p_s.boxer_id not in top_pose_per_boxer:
                    top_pose_per_boxer[p_s.boxer_id] = p_s
            candidates = list(top_pose_per_boxer.values())
        elif boxer_num == 1 and None not in boxer_id_set:
            frame_posescore_entities_sorted = sorted(frame_posescore_entities, key=lambda t: t.norm_dis_to_boxer_center)
            candidates = frame_posescore_entities_sorted[:3]
        else:
            frame_posescore_entities_sorted = sorted(frame_posescore_entities,
                                                     key=lambda t: t.points_scores_sum_after_re_pu)
            candidates = frame_posescore_entities_sorted[:3]

        need_fill = max(fill_to_least - len(candidates), 0)
        for i in range(need_fill):
            boxer_id = 10000 + i
            new_pose = self._pose_estimator.pose_type(np.zeros((25, 4), dtype=np.float32))
            candidates.append(self.PoseScore(new_pose, boxer_id, 0, 0, 0, 0, 1, False))

        return candidates

    @timeit
    def _get_top_n(self, frame_posescore_entities, n):
        # step1: get pose closest to every refer boxer
        # step2: sort by points socre sum, get top n
        # boxer_num = len({p_s.boxer_id for p_s in frame_posescore_entities})
        # if boxer_num == 1:
        #     top_n = sorted(frame_posescore_entities,key=lambda t:t.points_scores_sum_after_re_pu,reverse=True)[:n]
        # else:
        top_pose_per_boxer = self._getEachBoxerTop(frame_posescore_entities, fill_to_least=n)
        top_n = sorted(top_pose_per_boxer.values(), key=lambda t: t.points_scores_sum_after_re_pu, reverse=True)[:n]
        return top_n

    @timeit
    def _connectAndSmoothPoses(self, frame_posescore_entities, smooth):
        # assert len(frame_posescore_entities) >= 2,'Poses num should has been filled up to 2 or greater!'
        assert 0 < smooth <= 1, 'smooth value range: (0,1], 1 means not smooth'
        if self.prev_frame_poses_holder is None:
            self.prev_frame_poses_holder = self._get_top_n(frame_posescore_entities, 2)

            init_pose_inherit_times = np.zeros((25, 4), dtype=np.int)
            self.frame_poses_inherited_times = [init_pose_inherit_times, init_pose_inherit_times.copy()]
            self.frame_poses_fully_inherited_times = [[0], [0]]
            return self.prev_frame_poses_holder

        prev_pose1, prev_pose2 = self.prev_frame_poses_holder[:2, 0]
        candidates_poses = list(self._getCandidates(frame_posescore_entities, fill_to_least=2))
        poses_combination_distances = []
        for c_i, pose_entity in enumerate(candidates_poses):
            poses_combination_distances.append((0, c_i, pose_entity.pose.distance2(prev_pose1)))
            poses_combination_distances.append((1, c_i, pose_entity.pose.distance2(prev_pose2)))

        poses_combination_distances_sorted = sorted(poses_combination_distances, key=lambda t: t[2])
        logger.debug(f'poses_combination_distances_sorted: {poses_combination_distances_sorted}')
        _first_elem = poses_combination_distances_sorted.pop(0)
        _second_elem_base = filter(lambda t: t[0] != _first_elem[0] and t[1] != _first_elem[1],
                                   poses_combination_distances_sorted)
        try:
            _second_elem = next(_second_elem_base)
        except StopIteration:
            raise Exception('Hell, black night!')
            # _second_elem = ('-',)
        if _first_elem[0] == 0:
            cur_frame_poses_connected = [candidates_poses[_first_elem[1]], candidates_poses[_second_elem[1]]]
            poses_distances = [_first_elem[-1], _second_elem[-1]]
        else:
            cur_frame_poses_connected = [candidates_poses[_second_elem[1]], candidates_poses[_first_elem[1]]]
            poses_distances = [_second_elem[-1], _first_elem[-1]]

        # min_comb_dis = min(poses_combination_distances)
        # min_comb_dis_index = poses_combination_distances.index(min_comb_dis)
        # if min_comb_dis_index in [0,3]:
        #     cur_frame_poses_connected = frame_posescore_entities
        #     poses_distances = [poses_combination_distances[0],poses_combination_distances[3]]
        # else:
        #     cur_frame_poses_connected = frame_posescore_entities[::-1]
        #     poses_distances = [poses_combination_distances[2],poses_combination_distances[1]]

        if smooth != 1:
            logger.debug(f'poses_distances {poses_distances}')
            cur_frame_poses_connected = [self._smooth_poses(*z) for z in
                                         zip(self.prev_frame_poses_holder,
                                             cur_frame_poses_connected,
                                             [smooth, smooth],
                                             self.frame_poses_inherited_times,
                                             self.frame_poses_fully_inherited_times,
                                             [1, 1],
                                             poses_distances)]

        self.prev_frame_poses_holder = cur_frame_poses_connected
        return cur_frame_poses_connected

    def _smooth_poses(self, prev_pose_entity, cur_pose_entity, smooth_factor, inherited_times, fully_inherited_times,
                      inherit_times_thre, poses_distance):
        prev_pose, cur_pose = prev_pose_entity.pose, cur_pose_entity.pose
        cur_pose_zero_position = cur_pose.key_points == 0
        prev_pose_zero_position = prev_pose.key_points == 0
        torso_height = cur_pose_entity.pose.torso_height()
        poses_dis_thre = torso_height * 0.50
        # inherit_type: 0 not inherit; 1 partial inherit; 2 all inherit
        if poses_distance < poses_dis_thre:
            inherit_type = 1
            should_inherit_position = np.logical_and(cur_pose_zero_position,
                                                     inherited_times < inherit_times_thre)
            should_inherit_position[..., 3] = should_inherit_position[..., 2]  # is inherited flag column for plotting
        elif fully_inherited_times[0] < inherit_times_thre:
            inherit_type = 2
            should_inherit_position = np.ones(inherited_times.shape, dtype=np.bool)
            fully_inherited_times[0] += 1
            logger.debug(f'poses_distance {poses_distance} >= poses_dis_thre {poses_dis_thre}')
        else:
            inherit_type = 0
            should_inherit_position = np.zeros(inherited_times.shape, dtype=np.bool)
            logger.debug(f'prev_pose: {prev_pose.key_points} cur_pose: {cur_pose.key_points}')
            logger.debug(f'torso_height: {torso_height} x0.20: {poses_dis_thre} poses_distance: {poses_distance}')

        logger.debug(f'inherit_type: {inherit_type}')

        should_not_inherit_position = np.logical_not(should_inherit_position)
        if inherit_type != 2:
            fully_inherited_times[0] = 0
            inherited_times[should_inherit_position] += 1
            inherited_times[should_not_inherit_position] = 0

        cur_pose.key_points[should_inherit_position] = prev_pose.key_points[should_inherit_position]
        cur_pose.key_points[..., 3][should_inherit_position[..., 3]] = inherit_type
        if inherit_type == 0:
            return self.PoseScore(cur_pose, *cur_pose_entity[1:])

        cur_pose.key_points[..., :3][should_inherit_position[..., :3]] = prev_pose.key_points[..., :3][
            should_inherit_position[..., :3]]
        cur_pose_zero_position_after_inherit = cur_pose.key_points == 0
        cur_pose_after_smooth = smooth_factor * cur_pose.key_points + (1 - smooth_factor) * prev_pose.key_points
        should_not_smooth = np.zeros(inherited_times.shape, dtype=np.bool)
        should_not_smooth[prev_pose_zero_position] = True
        should_not_smooth[cur_pose_zero_position_after_inherit] = True
        should_not_smooth[..., 3] = True
        cur_pose_after_smooth[should_not_smooth] = cur_pose.key_points[should_not_smooth]
        return self.PoseScore(self._pose_estimator.pose_type(cur_pose_after_smooth), *cur_pose_entity[1:])


class BoxingAIImage(BoxingAI):
    def __init__(self, path):
        assert os.path.isfile(path)
        super().__init__()
        self.media = ImageCV(path)
        self.queue_name_img_tuple = Queue(maxsize=2)

    def _startProducingImgs(self):
        img = self.media.org()
        base_name = os.path.basename(self.media.fname)
        # self.require_show(img,title=f'{base_name}:org')
        img = self._execPrepHooks(img, base_name)
        # ImageCV(img,'testing').show()
        self.queue_name_img_tuple.put((base_name, img))
        self.queue_name_img_tuple.put('DONE')

    def _connectAndSmoothPoses(self, all_poses, smooth):
        return all_poses
