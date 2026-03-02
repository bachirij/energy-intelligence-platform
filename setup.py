from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="energy-intelligence-platform",  # package name
    version="0.1.0",                      # initial version
    packages=find_packages(),             # includes automatically all packages in the project
    install_requires=install_requires,   # depenencies from requirements.txt
    author="Jihane Bachiri",
    description="Hourly electricity demand forecasting using machine learning techniques",
    python_requires='>=3.7',              # minimum Python version requirement
)
