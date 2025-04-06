"""
setup.py for the fisheye_python python part of the project. Place
requirements in the install_requires list only if they are needed
by end-users, place development requirements in 'dev_requirements.txt'.
"""
from setuptools import setup

setup(
    name='fisheye',
    version='0.0.1',
    description='A library for calibrating and working with fisheye cameras.',
    license='TOOD',
    maintainer='Joseph Agrane',
    maintainer_email='josephagrane@gmail.com',
    author='Joseph Agrane',
    author_email='josephagrane@gmail.com',
    install_requires=[
        "numpy~=2.2.4",
        "opencv-python~=4.11.0.86",
    ]
)
