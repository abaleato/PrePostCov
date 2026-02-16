from setuptools import setup, find_packages

setup(
    name="prepostcov",
    version="0.1.0",
    description="Covariance and multipole tools for power spectrum and correlation function analysis.",
    author="Baleato Lizancos, Maus, White",
    author_email="a.baleatolizancos@berkeley.edu",
    url="https://github.com/abaleato/PrePostCov/tree/main",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "prepostcov=prepostcov.__main__:main"
        ]
    },
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
