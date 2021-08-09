from setuptools import setup

setup(
  name="ml-model-selection",
  version="0.0.2",
  py_modules =["model_selection"],
  description="Library for the selection of machine learning models",
  url="https://github.com/EnricoPittini/model-selection",
  author="Enrico Pittini",
  author_email="pittinienrico@hotmail.it",
  license="MIT",
  install_requires=['numpy',
                    'matplotlib',
                    'sklearn' ],
)
