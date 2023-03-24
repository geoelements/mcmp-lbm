from setuptools import setup, find_packages

setup(
    name='3D_MCMP_MRT_LBM',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'taichi',
        'vtk',
        'json'
    ],
    author='Qiuyu Wang',
    author_email='wangqiuyu@utexas.edu',
    description='A 3D multicomponent multiphase lattice Boltzmann solver with a Multi-Relaxation-Time collision scheme and a sparse storage structure',
    url='https://github.com/Amber1995/3D_MCMP_MRT_LBM',
)

