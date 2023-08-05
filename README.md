# 3D_MCMP_MRT_LBM
A 3D multicomponent multiphase lattice Boltzmann solver with a Multi-Relaxation-Time collision scheme and a sparse storage structure
The Multi-Relaxation-Time collision scheme and sparse storage structure can help improve the accuracy and efficiency of the simulations, allowing for larger and more complex simulations to be performed.

## Installation and running the code

1. Clone the repository using `git clone git@github.com:geoelements/mcmp-lbm.git`
2. Install dependencies using `poetry install` or `pip install -r requirements.txt`.
3. Specify input parameters in a JSON file `example.json`.
4. Run `python main.py path_to_input`, for example, `python main.py examples/example.json`

## Examples

Here are some examples of how to use this project:

1. Injection of water to granular media: `python main.py examples/hamburg_sand.json`
<img src="/screenshots/3d_hamburg_injection.gif" alt="Screenshot" width="500"/>

2. A droplet test: `python main.py examples/droplet.json`
<img src="/screenshots/droplet_test.gif" alt="Screenshot" width="500"/>

3. Determination contact angle for one fluid surrounded by another fluid vs. adhesion parameters:
<img src="/screenshots/contact_angle.png" alt="Screenshot" width="500"/>

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

If you encounter any problems while using this project, please [open an issue](https://github.com/Amber1995/3D-MCMP-MRT-LBM/issues/new).

## Contributors

* Qiuyu Wang
* Krishna Kumar

