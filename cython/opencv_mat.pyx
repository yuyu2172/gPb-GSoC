import numpy as np
cimport numpy as np  # for np.ndarray
from libc.string cimport memcpy
from opencv_mat cimport *
# from contour2ucm cimport contour2ucm

from libcpp cimport vector



cdef Mat np2Mat2D(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"

    cdef np.ndarray[np.float32_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.float32)
    cdef float* im_buff = <float*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_32FC1)
    memcpy(m.data, im_buff, r*c)
    return m


cdef object Mat2np(Mat m):
    # Create buffer to transfer data from m.data
    cdef Py_buffer buf_info
    # Define the size / len of data
    cdef size_t len = m.rows * m.cols * m.channels() * sizeof(CV_32FC1)
    # Fill buffer
    PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)
    # Get Pyobject from buffer data
    Pydata  = PyMemoryView_FromBuffer(&buf_info)

    # Create ndarray with data
    shape_array = (m.rows, m.cols, m.channels())
    ary = np.ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=np.float32)

    # # BGR -> RGB
    # ary = np.dstack((ary[...,2], ary[...,1], ary[...,0]))
    # Convert to numpy array
    pyarr = np.asarray(ary)
    return pyarr


def np2Mat2np(nparray):
    cdef Mat m
    # Convert numpy array to cv::Mat
    m = np2Mat2D(nparray)
    # Convert cv::Mat to numpy array
    pyarr = Mat2np(m)
    return pyarr


cpdef contour2ucm_py(np.ndarray pb, np.ndarray pb_ori):
    cdef vector.vector[Mat] pb_ori_vec
    cdef int i
    cdef Mat output
    cdef Mat temp

    for i in range(8):
        temp = np2Mat2D(pb_ori[i])
        pb_ori_vec.push_back(temp)
    contour2ucm(np2Mat2D(pb), pb_ori_vec, output, 0)
    return Mat2np(output)
