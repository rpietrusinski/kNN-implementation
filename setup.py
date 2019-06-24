import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myknn",
    version="0.0.3",
    author="Robert",
    author_email="pietrusinski.robert@gmail.com",
    description="Knn implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpietrusinski/kNN-implementation",
    packages=["myknn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
