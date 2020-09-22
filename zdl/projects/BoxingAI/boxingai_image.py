import os
from multiprocessing.queues import Queue

from zdl.utils.media.image import ImageCV

from zdl.projects.BoxingAI.boxingai import BoxingAI


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
