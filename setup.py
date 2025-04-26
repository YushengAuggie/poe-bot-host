"""
Setup script for Poe Bots Framework.
This is mainly for compatibility with older package managers.
Modern Python packaging should use pyproject.toml instead.
"""

from setuptools import find_packages, setup

setup(
    name="poe_bots",
    version="1.0.0",
    author="Poe Bots Framework Team",
    description="A framework for creating, testing, and deploying bots for the Poe platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/poe-bots-framework/poe-bots",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi-poe>=0.0.21",
        "modal-client>=0.52.4271",
        "python-dotenv>=1.0.0",
        "uvicorn>=0.17.6",
        "fastapi>=0.105.0",
        "pydantic>=2.0.0",
        "requests>=2.27.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.1.0",
            "ruff>=0.0.65",
            "pyright>=1.1.300",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
