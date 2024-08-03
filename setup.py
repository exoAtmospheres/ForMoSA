from setuptools import setup, find_packages, Extension
import re


with open("README.md", "r") as fh:
    long_description = fh.read()


# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

description = '''Welcome to ForMoSA, an open-source Python package. 
We designed this tool to model exoplanetary atmospheres with a forward modeling approach. 

We encourage the community to exploit its capabilities!'''

setup(name='ForMoSA',
      version=get_property("__version__", "ForMoSA"),
      description='ForMoSA: Forward Modeling Tool for Spectral Analysis',
      long_description=description,
      url='https://github.com/exoAtmospheres/ForMoSA',
      author='P. Palma-Bifani, S. Petrus, M. Ravet, A. Denis, M. Bonnefoy, G. Chauvin',
      author_email='paulina.palma-bifani@oca.eu',
      license='BSD 3-Clause License',
      packages=['ForMoSA', 'ForMoSA.adapt', 'ForMoSA.nested_sampling', 'ForMoSA.plotting'],
      install_requires=get_requires(),
      include_package_data = True,
      zip_safe=False
      )