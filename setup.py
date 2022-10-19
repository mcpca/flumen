from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'torch>=1.8.0',
]

with open('README.md', 'r') as f:
    readme = f.read()


setup(name='ds-nn-representations',
      version='0.1.0',
      description='A neural architecture for dynamical systems with inputs',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/mcpca/ds-nn-representations',
      author='Miguel Aguiar',
      author_email='aguiar@kth.se',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.6',
)
