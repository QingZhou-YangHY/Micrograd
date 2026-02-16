import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd_test",
    version="0.1.0",
    author="Huanyu Yang",
    author_email="yhy3868850350@gmail.com",
    description="A tiny autograd engine implementation based on [Andrej Karpathy's tutorial](https://github.com/karpathy/micrograd)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QingZhou-YangHY/Micrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)