#!/usr/bin/env python
# coding: utf-8
import taichi as ti
import numpy as np
import math
from sympy import inverse_mellin_transform
from pyevtk.hl import gridToVTK
import pandas as pd
import vtk
import time
import LBM_parameters as lbm
import json
import os, sys
import LBM_solver as LBM

# ti.init(arch=ti.gpu, dynamic_index=False)
ti.init(arch=ti.cpu,cpu_max_num_threads=32, dynamic_index=False)

if len(sys.argv) < 2:
    print("Please provide input file path.")
    sys.exit(1)

input_file = sys.argv[1]
LBM.read_json(input_file)

for x in range(LBM.lx):
    for y in range(LBM.ly):
        LBM.solid_np[x][y][LBM.lz - 1] = 1
        LBM.solid_np[x][y][0] = 1
LBM.nb_solid = np.count_nonzero(LBM.solid_np)

if LBM.MC:
    print("Run multi-compont solver")
    lbm_solver = LBM.D3Q19_MC(sparse_mem=LBM.sparse)
    print("{} fluid nodes have been activated!".format(lbm_solver.activity_checking()))
    print(
        "{} fluid nodes in the computational domain!".format(
            LBM.lx * LBM.ly * LBM.lz - LBM.nb_solid
        )
    )
else:
    print("Run single-compont solver")
    lbm_solver = LBM.D3Q19_SC(sparse_mem=LBM.sparse)

lbm_solver.init_field()
lbm_solver.run(LBM.nb_steps)

