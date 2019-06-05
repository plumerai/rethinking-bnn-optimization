from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="rethinking-bnn-optimization",
    version="0.1.0",
    author="Plumerai",
    author_email="koen@plumerai.com",
    description='Implementation for paper "Latent Weights Do Not Exist: Rethinking Binary Neural Network Optimization"',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/rethinking-bnn-optimization.git",
    packages=find_packages(),
    python_requires=">=3.6",
    license="Apache 2.0",
    install_requires=[
        "click==7.0",
        "tensorflow-datasets==1.0.2",
        "larq==0.2.0",
        "zookeeper==0.1.1",
    ],
    extras_require={
        "tensorflow": ["tensorflow==1.14.0rc0"],
        "tensorflow_gpu": ["tensorflow-gpu==1.14.0rc0"],
    },
    entry_points="""
        [console_scripts]
        bnno=bnn_optimization.train:cli
    """,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
