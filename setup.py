from setuptools import setup, find_packages

setup(
    name="RRN",
    version="0.1.0",
    author="Youness Karafi",
    author_email="youness.karafi@hotmail.fr;youness_karafi@um5.ac.ma",
    description="Reparametrization and Regression Network (RRN) for simultaneous dimensionality reduction and surrogate modeling",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/youkas/RRN",
    project_urls={
        "Paper": "https://www.sciencedirect.com/science/article/pii/S0957417425022432",
    },
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Dimension Reduction :: Surrogate Modeling",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.22.0",
        "tensorflow>=2.12.1",
        "pymoo>=0.5.0",
    ],
)
