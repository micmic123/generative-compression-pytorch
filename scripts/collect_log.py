import os
from datetime import datetime
from distutils.dir_util import copy_tree


logdir='../results'
target = [
    'm00_s1_5050_q0',
    'm00_s1_7525_q0',
    'm00_s1_9010_q0',
    'm0001_s1_5050_q0',
    'm00_s2_5050_q0',
    'm00_s2_7525_q0',
    'm00_s1_5050_q1',
    'm00_s1_7525_q1',
]
destdir = os.path.join('../logs/', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
paths = [(t, os.path.join(logdir, t, 'logs')) for t in target]

for t, path in paths:
    copy_tree(path, os.path.join(destdir, t))
print(destdir)

