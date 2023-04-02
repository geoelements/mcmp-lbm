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
import tifffile

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
                    
def read_positions(position_filename,lx,ly,lz,dim,scale_factor):
    solid_np = np.zeros((lx, ly, lz), dtype=np.int8)
    with open(position_filename) as f:
        Lines = f.readlines()
    if dim ==3:
        i = 0
        for line in Lines:
            i += 1
            k = float(line)/scale_factor
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
            k = float(line)/scale_factor
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

def read_tiff(tiff_path,lx,ly,lz,scale_factor):
    
    print("Reading trinarized image "+tiff_path)
    trinarized_00 = tifffile.imread(tiff_path)
    print("Done.")

    xmin = int(trinarized_00.shape[0]/2-lx/2*scale_factor)
    ymin = int(trinarized_00.shape[1]/2-ly/2*scale_factor)
    zmin = int(trinarized_00.shape[2]/2-lz/2*scale_factor)
    xmax = xmin + lx*scale_factor
    ymax = ymin + ly*scale_factor
    zmax = zmin + lz*scale_factor

    print("Box dimension: xmin = {:.2f}, xmax = {:.2f}".format(xmin,xmax))
    print("               ymin = {:.2f}, ymax = {:.2f}".format(ymin,ymax))
    print("               zmin = {:.2f}, zmax = {:.2f}".format(zmin,zmax))

    grains = trinarized_00[xmin:xmax:scale_factor,ymin:ymax:scale_factor,zmin:zmax:scale_factor]
    # grain_index = np.unique(grains)
    # print(grains.shape,grain_index.size)
    solid_np = np.sign(grains)

    return solid_np

def Press(rho_value,carn_star,T_Tc,G_1s):
    if carn_star: 
        a = 1.0
        b = 4.0
        R = 1.0
        Tc = 0.0943

        T = T_Tc * Tc
        eta = b * rho_value / 4.0
        eta2 = eta * eta
        eta3 = eta2 * eta
        rho2 = rho_value * rho_value
        one_minus_eta = 1.0 - eta
        one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta
        return (
            rho_value * R * T * (1 + eta + eta2 - eta3) / one_minus_eta3 - a * rho2
        )
    else:
        cs2 = 1.0 / 3.0
        psi = 1.0 - np.exp(-rho_value)
        psi2 = psi * psi
        return cs2 * rho_value + cs2 * G_1s / 2 * psi2

    