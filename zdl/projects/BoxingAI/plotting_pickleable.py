import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib import animation
from matplotlib.artist import Artist
from zdl.utils.io.log import logger


class PlottingPickleable:
    def __init__(self, media_info, model_pose):
        self.media_info = media_info
        self.model_pose = model_pose
        self.frames_contents = {}

        self.frames_artists = []

        self._initFig()
        self._initPose()

    def _initPose(self):
        if self.model_pose == 'BODY_25':
            self.pose_sections = [[17, 15, 0, 16, 18], [0, 1, 8], [1, 2, 3, 4], [1, 5, 6, 7], [8, 9, 10, 11],
                                  [8, 12, 13, 14], [11, 23, 22, 11, 24], [14, 20, 19, 14, 21]]
        elif self.model_pose == 'BODY_25B':
            self.pose_sections = [[4, 2, 0, 1, 3], [18, 17, 6, 12, 11, 5, 17], [6, 8, 10], [5, 7, 9], [12, 14, 16],
                                  [11, 13, 15], [16, 22, 23, 16, 24], [15, 19, 20, 15, 21]]
        else:
            raise NotImplementedError()
        self.pose_colors = ['blue', 'red', 'gray']

    def _initFig(self):
        w, h = self.media_info['width'], self.media_info['height']
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
        self.fig = fig
        self.ax_main = ax_main
        self.ax_tip = ax_tip

    def newFrame(self, poses=[], points=[], texts=[], rects=[]):
        self.frames_contents[len(self.frames_contents)] = {'poses': poses, 'points': points, 'texts': texts,
                                                           'rects': rects}

    def _frame2artists(self, i) -> List[Artist]:
        poses = self.frames_contents[i]['poses']
        points = self.frames_contents[i]['points']
        texts = self.frames_contents[i]['texts']
        rects = self.frames_contents[i]['rects']

        line_objs = []
        if poses:
            for p_i, pose in enumerate(poses):
                pose_nonzero_b = pose.keyPoints != 0
                pose_x_or_y_nonzero_b = np.logical_or(pose_nonzero_b[..., 0], pose_nonzero_b[..., 1])
                for i, s in enumerate(self.pose_sections):
                    s_nonzero = np.asarray(s)[pose_x_or_y_nonzero_b[s]]
                    x_a = pose.keyPoints[s_nonzero][..., 0]
                    y_a = pose.keyPoints[s_nonzero][..., 1]
                    # section = ([person_trans[0][j] for j in s if person_trans[0][j]!=0],
                    #         [person_trans[1][j] for j in s if person_trans[1][j]!=0])
                    markersize = 6 if i in [0, 6, 7] else 8
                    lobj, = self.ax_main.plot(x_a, y_a, ls='-', color=self.pose_colors[p_i], marker='o',
                                              markersize=markersize, markerfacecolor='white')
                    line_objs.append(lobj)

                # set marker of inherited points to yellow
                # Log.debug(pose[3])
                partially_inherited_points = pose.keyPoints[..., 3] == 1
                should_plot = np.logical_and(partially_inherited_points, pose_x_or_y_nonzero_b)
                lobj, = self.ax_main.plot(pose.keyPoints[..., 0][should_plot], pose.keyPoints[..., 1][should_plot],
                                          ls='', color=self.pose_colors[p_i], marker='o', markersize=6,
                                          markerfacecolor=self.pose_colors[p_i])
                line_objs.append(lobj)
                fully_inherited_points = pose.keyPoints[..., 3] == 2
                should_plot = np.logical_and(fully_inherited_points, pose_x_or_y_nonzero_b)
                lobj, = self.ax_main.plot(pose.keyPoints[..., 0][should_plot], pose.keyPoints[..., 1][should_plot],
                                          ls='', color=self.pose_colors[p_i], marker='+', markersize=10,
                                          markerfacecolor=self.pose_colors[p_i])
                line_objs.append(lobj)

        point_objs = []
        if points:
            if isinstance(points, list):
                points = np.asarray(points)
            # Log.debug(points)
            # plot points by line
            lobj, = self.ax_main.plot(points[:, 0], points[:, 1], ls='', color='white', marker='o', markersize=10,
                                      markerfacecolor='black')
            point_objs.append(lobj)

        rect_objs = []
        if rects:
            for i, r in enumerate(rects):
                rect = plt.Rectangle((r[1], r[0]),
                                     r[3] - r[1],
                                     r[2] - r[0], fill=False,
                                     edgecolor='#e1f2e9', linewidth=2.5)
                rect_objs.append(self.ax_main.add_patch(rect))

        text_objs = []
        if texts.get('main'):
            text_objs.append(self.ax_tip.text(15, 15, ' '.join(texts['main']), verticalalignment='top',
                                              fontsize=18, color='black'))
        ax_tip_half_w = self.ax_tip.get_xlim()[1] / 2
        boxer1_left_margin, boxer2_left_margin, boxer_top_margin = 20, 20 + ax_tip_half_w, 70
        if texts.get('boxer1'):
            text_objs.append(self.ax_tip.text(boxer1_left_margin, boxer_top_margin, '\n'.join(texts['boxer1']),
                                              verticalalignment='top', fontsize=12, color=self.pose_colors[0]))
        if texts.get('boxer2'):
            text_objs.append(self.ax_tip.text(boxer2_left_margin, boxer_top_margin, '\n'.join(texts['boxer2']),
                                              verticalalignment='top', fontsize=12, color=self.pose_colors[1]))

        return line_objs + point_objs + rect_objs + text_objs

    def plotFrame(self, i):
        artists = self._frame2artists(i)
        for artist in artists:
            artist.draw(self.ax_main)

    # main entry
    def animate(self, save_video_to=None, plot_center=False, plot_rect=False):
        logger.debug('animating...')
        fps = self.media_info['fps']
        for i in range(len(self.frames_contents)):
            self.frames_artists.append(self._frame2artists(i))
        anim = animation.ArtistAnimation(self.fig, self.frames_artists, interval=1000 / fps, blit=True,
                                         repeat_delay=1000)
        # anim.save('anim-.mp4')
        plt.show()

    @classmethod
    def load(cls, fname) -> 'PlottingPickleable':
        with open(fname, 'rb') as f:
            self_obj = pickle.load(f)  # type:'PlottingPickleable'
        self_obj._initFig()
        self_obj._initPose()
        return self_obj

    def dump(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        state = {
            'media_info': self.media_info,
            'model_pose': self.model_pose,
            'frames_contents': self.frames_contents,
            'frames_artists': self.frames_artists,
            # 'pose_sections': self.pose_sections,
            # 'pose_colors': self.pose_colors,
            # 'ax_main': self.ax_main,
            # 'ax_tip': self.ax_tip,
            # 'fig': self.fig
        }
        return state

    def __setstate__(self, state):
        self.media_info = state['media_info']
        self.model_pose = state['model_pose']
        self.frames_contents = state['frames_contents']
        self.frames_artists = state['frames_artists']
        # self.pose_sections = state['pose_sections']
        # self.pose_colors = state['pose_colors']
        # self.ax_main = state['ax_main']
        # self.ax_tip = state['ax_tip']
        # self.fig = state['fig']
