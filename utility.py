#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import style
from pylab import *
import os
import math
import vtk 
from vtk.util import numpy_support

def place_sphere(x, y, z, R,lx,ly,lz,solid_np):
    xmin = x - R
    ymin = y - R
    zmin = z - R
    xmax = x + R
    ymax = y + R
    zmax = z + R
    for px in range(xmin, xmax + 1):
        for py in range(ymin, ymax + 1):
            for pz in range(zmin, zmax + 1):
                dx = px - x
                dy = py - y
                dz = pz - z
                dist2 = dx * dx + dy * dy + dz * dz
                R2 = R * R
                if dist2 < R2:
                    near_px = (
                        math.floor(px + 0.5)
                        if math.floor(px + 0.5)
                        else math.floor(px + 0.5) + lx
                    )
                    near_py = (
                        math.floor(py + 0.5)
                        if math.floor(py + 0.5)
                        else math.floor(py + 0.5) + ly
                    )
                    near_pz = (
                        math.floor(pz + 0.5)
                        if math.floor(pz + 0.5)
                        else math.floor(pz + 0.5) + lz
                    )
                    if near_px >= lx:
                        near_px -= lx
                    if near_py >= ly:
                        near_py -= ly
                    if near_pz >= lz:
                        near_pz -= lz
                    solid_np[near_px, near_py, near_pz] = 1

def place_column(x, y, R,lx,ly,lz,solid_np):
    xmin = x - R
    ymin = y - R
    xmax = x + R
    ymax = y + R
    for px in range(xmin, xmax + 1):
        for py in range(ymin, ymax + 1): 
            dx = px - x
            dy = py - y
            dist2 = dx * dx + dy * dy
            R2 = R * R
            if dist2 < R2:
                near_px = (
                    math.floor(px + 0.5)
                    if math.floor(px + 0.5)
                    else math.floor(px + 0.5) + lx
                )
                near_py = (
                    math.floor(py + 0.5)
                    if math.floor(py + 0.5)
                    else math.floor(py + 0.5) + ly
                )
                if near_px >= lx:
                    near_px -= lx
                if near_py >= ly:
                    near_py -= ly
                for z in range(lz):
                    solid_np[near_px, near_py, z] = 1
                    
def read_positions(position_filename,lx,ly,lz,dim):
    solid_np = np.zeros((lx, ly, lz), dtype=np.int8)
    with open(position_filename) as f:
        Lines = f.readlines()
    if dim ==3:
        i = 0
        for line in Lines:
            i += 1
            k = float(line)
            k = int(k)
            if i == 1:
                x = k
            elif i == 2:
                y = k
            elif i == 3:
                z = k
            else:
                i = 0
                r = k
                place_sphere(x, y, z, r,lx,ly,lz,solid_np)

    elif dim ==2:
        i = 0
        for line in Lines:
            i += 1
            k = float(line)
            k = int(k)
            if i == 1:
                x = k
            elif i == 2:
                y = k
            else:
                i = 0
                r = k
                place_column(x, y, r,lx,ly,lz,solid_np)
            
    return solid_np


                    
