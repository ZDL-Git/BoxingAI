import os
import time
from multiprocessing.context import Process
from multiprocessing.queues import Queue
from typing import Union, Tuple, List, Dict

import numpy as np
import psutil
from zdl.utils.io.log import logger
from zdl.utils.media.video import Video
from zdl.utils.time.counter import timeit

from zdl.projects.BoxingAI.boxingai import BoxingAI


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

    def setShow(self, show: bool):
        if show:
            num_to_show = len(self.indices) if self.indices else self.media.getInfo()['frame_c']
            if num_to_show > 10 and input(f"About {num_to_show} pictures to be displayed, which may lead to crash."
                                          " Do you want to display anyway? [y/n]:") != 'y':
                show = False
        self.show = show
        return self

    def _startProducingImgs(self):
        # _newProcessToSeedVideoFrames:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        logger.debug(f'parent pid: {current_process.pid}')
        logger.debug(f'children pids: {[c.pid for c in children]}')

        def _loadVideo(path, indices, queue):
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

        pw = Process(target=_loadVideo, args=(self.media.fname, self.indices, self.queue_name_img_tuple),
                     name='imgs_producer_process')
        pw.daemon = True
        pw.start()

    def _getEachBoxerTop(self, frame_posescore_entities, fill_to_least) -> Dict[int, BoxingAI.PoseScore]:
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
    def _getTopN(self, frame_posescore_entities, n) -> List[BoxingAI.PoseScore]:
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
            self.prev_frame_poses_holder = self._getTopN(frame_posescore_entities, 2)

            init_pose_inherit_times = np.zeros((25, 4), dtype=np.int)
            self.frame_poses_inherited_times = [init_pose_inherit_times, init_pose_inherit_times.copy()]
            self.frame_poses_fully_inherited_times = [[0], [0]]
            return self.prev_frame_poses_holder

        prev_pose1, prev_pose2 = [pose_score.pose for pose_score in self.prev_frame_poses_holder[:2]]
        candidates_poses = list(self._getCandidates(frame_posescore_entities, fill_to_least=2))
        poses_combination_distances = []
        for c_i, pose_entity in enumerate(candidates_poses):
            poses_combination_distances.append((0, c_i, pose_entity.pose.distanceTo(prev_pose1)))
            poses_combination_distances.append((1, c_i, pose_entity.pose.distanceTo(prev_pose2)))

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
            cur_frame_poses_connected = [self._smoothPoses(*z) for z in
                                         zip(self.prev_frame_poses_holder,
                                             cur_frame_poses_connected,
                                             [smooth, smooth],
                                             self.frame_poses_inherited_times,
                                             self.frame_poses_fully_inherited_times,
                                             [1, 1],
                                             poses_distances)]

        self.prev_frame_poses_holder = cur_frame_poses_connected
        return cur_frame_poses_connected

    def _smoothPoses(self, prev_pose_entity, cur_pose_entity, smooth_factor, inherited_times, fully_inherited_times,
                     inherit_times_thre, poses_distance):
        prev_pose, cur_pose = prev_pose_entity.pose, cur_pose_entity.pose
        cur_pose_zero_position = cur_pose.key_points == 0
        prev_pose_zero_position = prev_pose.key_points == 0
        torso_height = cur_pose_entity.pose.torsoHeight()
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
