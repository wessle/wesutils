import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wesutils",
    version="0.0.5",
    author="Wesley Suttle",
    description="my most commonly used utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wessle/wesutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'pyyaml'
    ]
)
