#!/usr/bin/env python
# coding: utf-8
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
from matplotlib import style
from pylab import *
import os
import math
import vtk 
from vtk.util import numpy_support

# Predefined compound types
i32_vec3d = ti.types.vector(3, ti.i32)
f32_vec3d = ti.types.vector(3, ti.f32)
f32_vec2d = ti.types.vector(2, ti.f32)

M_np = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, -2, 2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, -2, 2, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
        [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
        [0, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
        [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1],
    ]
)

inv_M_np = np.linalg.inv(M_np)
reversed_e = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
)

cs2 = 1.0 / 3.0

"""Definition of 3D LBM weights"""
t0 = 1.0 / 3.0
t1 = 1.0 / 18.0
t2 = 1.0 / 36.0
t = np.array(
    [t0, t1, t1, t1, t1, t1, t1, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2]
)

"""Definition of Shan-Chen factors for force computation"""
w0 = 0
w1 = 2
w2 = 1
w = np.array(
    [t0, t1, t1, t1, t1, t1, t1, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2]
)


# x component of predefined velocity in Q directions
e_xyz_list = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
]

def update_S(niu_l,niu_g):
    tau_lf = 3.0 * niu_l + 0.5
    tau_gf = 3.0 * niu_g + 0.5

    # Diagonal relaxation matrix for fluid 1
    s_lv = 1.0 / tau_lf
    s_lother = 8.0 * (2.0 - s_lv) / (8.0 - s_lv)
    S_dig_np_l = np.array(
        [
            1,
            s_lv,
            s_lv,
            1,
            s_lother,
            1,
            s_lother,
            1,
            s_lother,
            s_lv,
            s_lv,
            s_lv,
            s_lv,
            s_lv,
            s_lv,
            s_lv,
            s_lother,
            s_lother,
            s_lother,
        ]
    )
    
    # Diagonal relaxation matrix for fluid 2
    s_gv = 1.0 / tau_gf
    s_gother = 8.0 * (2.0 - s_gv) / (8.0 - s_gv)
    S_dig_np_g = np.array(
        [
            1,
            s_gv,
            s_gv,
            1,
            s_gother,
            1,
            s_gother,
            1,
            s_gother,
            s_gv,
            s_gv,
            s_gv,
            s_gv,
            s_gv,
            s_gv,
            s_gv,
            s_gother,
            s_gother,
            s_gother,
        ]
    )
    
    return tau_lf,tau_gf,S_dig_np_l,S_dig_np_g

    