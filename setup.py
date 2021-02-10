import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'tensorflow>=2.3',
    'tensorflow_addons>=0.10'
]

test_require = [
    'pytest',
    'numpy'
]

setuptools.setup(
    name="tf_clahe",
    version="0.0.2",
    author="Isaac Sears",
    author_email="is6gc@virginia.edu",
    description="Contrast limited adaptive histogram equalization implemented in TF ops",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isears/tf_clahe",
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=install_requires,
    test_require=test_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
