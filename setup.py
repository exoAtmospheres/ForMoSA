from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='ForMoSA',
      version='1.0.0',
      description='ForMoSA: Forward Modeling Tool for Spectral Analysis',
      url='https://github.com/exoAtmospheres/ForMoSA',
      author='P. Palma-Bifani, S. Petrus',
      author_email='paulina.palma-bifani@oca.eu',
      license='BSD 2-Clause License',
      packages=['ForMoSA'],
      install_requires=[
        'astropy<5',
        'configobj',
        'corner',
        'extinction', 
        'nestle',
        'matplotlib',
      	'numpy', 
        'PyAstronomy',
        'scipy',
        'spectres',
        'xarray'],
      include_package_data = True,
      zip_safe=False,
      python_requires='>=3.7')