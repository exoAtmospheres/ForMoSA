from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

setup(name='ForMoSA',
      version='1.1.0',
      description='ForMoSA: Forward Modeling Tool for Spectral Analysis',
      url='https://github.com/exoAtmospheres/ForMoSA',
      author='P. Palma-Bifani, S. Petrus, M. Ravet, M. Bonnefoy, G. Chauvin',
      author_email='paulina.palma-bifani@oca.eu',
      license='BSD 2-Clause License',
      packages=['ForMoSA', 'ForMoSA.adapt', 'ForMoSA.nested_sampling', 'ForMoSA.plotting', 'ForMoSA.phototeque'],
      install_requires=get_requires(),
      include_package_data = True,
      zip_safe=False
      )