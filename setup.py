from setuptools import setup, find_packages

setup(
    name="SMSOceanMeshToolkit",
    version="0.1.0",
    packages=find_packages(),
    author="Keith Roberts",
    author_email="kroberts@baird.com",
    description="Tools for oceanmesh to interface with SMS",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/krober10nd/SMSOceanMeshToolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'geopandas',
        'shapely',
        'matplotlib',
        'pyproj',
        'fiona',
        'scipy',
        'xarray',
        'inpoly @ git+https://github.com/dengwirda/inpoly-python.git@v0.1.2',
    ],
)
