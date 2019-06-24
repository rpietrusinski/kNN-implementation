import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kNN-implementation",
    version="0.0.1",
    author="Robert",
    author_email="pietrusinski.robert@gmail.com",
    description="Knn implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpietrusinski/kNN-implementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
