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
import lbm_parameters as parameters
import json
import os, sys
import lbm_solver as lbm

# ti.init(arch=ti.gpu, dynamic_index=False)
ti.init(arch=ti.cpu,cpu_max_num_threads=32)

if len(sys.argv) < 2:
    print("Please provide input file path.")
    sys.exit(1)

input_file = sys.argv[1]
lbm.read_json(input_file)

if lbm.MC:
    print("Run multi-compont solver")
    solver = lbm.D3Q19_MC(sparse_mem=lbm.sparse)
else:
    print("Run single-compont solver")
    solver = lbm.D3Q19_SC(sparse_mem=lbm.sparse)

solver.init_field()
solver.run(lbm.nb_steps)

