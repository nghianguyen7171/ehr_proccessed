from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ehr-datasets",
    version="1.0.0",
    author="EHR Datasets Contributors",
    author_email="",
    description="EHR datasets preprocessing scripts for time-series prediction and tabular data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ehr-datasets",  # Update with actual repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.17",
        ],
        "gpu": [
            "cupy-cuda11x>=10.6.0",  # For CUDA 11.x
            # "cupy-cuda12x>=11.0.0",  # For CUDA 12.x - uncomment if needed
        ],
        "advanced": [
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
            "bokeh>=2.4.0",
            "altair>=4.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line tools here if needed
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.csv", "*.txt", "*.md"],
    },
)


