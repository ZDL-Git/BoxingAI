from collections import namedtuple

PoseScore = namedtuple('PoseScore', ['pose', 'boxer_id', 'completion_score', 'completion_multi_boxerScore',
                                     'points_score_sum', 'points_scores_sum_after_re_pu',
                                     'norm_dis_to_boxer_center',
                                     'knee_and_below_nonzero_exists'])
