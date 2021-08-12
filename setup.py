from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name="ml-model-selection",
  version="0.0.4",
  py_modules =["model_selection"],
  description="Library for the selection of machine learning models",
  long_description=long_description,
  long_description_content_type='text/markdown',
  url="https://github.com/EnricoPittini/model-selection",
  author="Enrico Pittini",
  author_email="pittinienrico@hotmail.it",
  license="MIT",
  install_requires=['numpy',
                    'matplotlib',
                    'sklearn' ],
  classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
