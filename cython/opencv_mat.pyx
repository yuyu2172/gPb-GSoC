import numpy as np
cimport numpy as np  # for np.ndarray
from libc.string cimport memcpy
from opencv_mat cimport *
# from contour2ucm cimport contour2ucm

from libcpp cimport vector

import ctypes as C

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef Mat np2Mat2D_float32(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"

    cdef np.ndarray[np.float32_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.float32)
    cdef float* im_buff = <float*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_32FC1)
    memcpy(m.data, im_buff, r*c)
    return m


cdef Mat np2Mat2D_int32(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"

    cdef np.ndarray[np.int32_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.int32)
    cdef int* im_buff = <int*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_32SC1)
    memcpy(m.data, im_buff, r * c)
    return m


cdef object Mat2np_int32(Mat m):
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> m.rows
    shape[1] = <np.npy_intp> m.cols
    
    # http://gael-varoquaux.info/programming/
    # cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html
    arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_INT32, m.data)
    return arr


cdef object Mat2np_float32(Mat m):
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> (m.rows * m.cols)
    arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT32, m.data)
    return arr


def np2Mat2np_float32(nparray):
    cdef Mat m
    # Convert numpy array to cv::Mat
    m = np2Mat2D_float32(nparray)
    # Convert cv::Mat to numpy array
    pyarr = Mat2np_float32(m)
    return pyarr


def np2Mat2np_int32(nparray):
    cdef Mat m
    # Convert numpy array to cv::Mat
    m = np2Mat2D_int32(nparray)
    # Convert cv::Mat to numpy array
    pyarr = Mat2np_int32(m)
    return pyarr


# cpdef contour2ucm_py(np.ndarray pb, np.ndarray pb_ori):
#     cdef vector.vector[Mat] pb_ori_vec
#     cdef int i
#     cdef Mat output
#     cdef Mat temp
# 
#     for i in range(8):
#         temp = np2Mat2D(pb_ori[i])
#         pb_ori_vec.push_back(temp)
#     contour2ucm(np2Mat2D(pb), pb_ori_vec, output, 0)
#     return Mat2np(output)
