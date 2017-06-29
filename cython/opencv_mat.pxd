cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

# For cv::Mat usage
cdef extern from "core/core.hpp":
    cdef int  CV_WINDOW_AUTOSIZE
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_32SC1
    cdef int CV_32FC1

cdef extern from "core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void* data
        int rows
        int cols
        int channels()

    cdef cppclass Point:
        int x
        int y


# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

# cdef Mat np2Mat(np.ndarray ary)
# 
# cdef np.ndarray Mat2np(Mat mat)


##############################################################################
# For this project
##############################################################################
cdef extern from "contour2ucm.h":
    cdef cppclass contour_vertex:
        int id
        bint is_subdivision
        int x
        int y
        vector[contour_edge] edges_start
        vector[contour_edge] edges_end

        Point point()
        
    cdef cppclass contour_edge:
        int id
        int contour_equiv_id
        vector[int] x_coords
        vector[int] y_coords
        contour_vertex * vertex_start
        contour_vertex * vertex_start
        int vertex_start_enum
        int vertex_end_enum

        int size()
        double length()
        Point point()

    cdef void contour2ucm(Mat&, vector[Mat]&, Mat&, bint)
