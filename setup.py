from setuptools import setup
from setuptools import find_packages

setup(
    name='ggrn_backend3',
    version='0.1.0',
    description='ML',
    #url
    author='Eric Kernfeld',
    author_email='eric.kern13@gmail.com',
    packages=find_packages(),
    install_requires=['pytorch', 'pytorch-lightning','numpy','scikit-learn','pandas','anndata']
)