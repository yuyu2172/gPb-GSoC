from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(
        'contour2ucm',
        [
            'contour2ucm.cpp',
            'watershed.cpp',
            'opencv_mat.pyx',
            'ucm_mean_pb.cpp',
        ],
        libraries=[
            'opencv_core',
            'opencv_videoio',
            'opencv_highgui',
            'opencv_imgproc'
        ],
        language="c++",
        extra_link_args=['-L/usr/lib/x86_64-linux-gnu/', '-g'],
        include_dirs=[
            np.get_include(),
            '/usr/include/opencv2',
            '/usr/include/opencv',
            '.',
        ],
        extra_compile_args=["-w", '-g'],
    )]
)
