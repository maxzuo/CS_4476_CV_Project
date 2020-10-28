import setuptools

setuptools.setup(
    name="tracker",
    version="0.0.1",
    author="Max Zuo",
    author_email="max.zuo@gmail.com",
    description="Wrapper convenience methods for OpenCV tracker objects, along with a tracking model zoo",
    long_description="None",
    url="https://github.com/maxzuo/CS_4476_CV_Project/tree/tracker",
    packages=setuptools.find_packages(),
    install_requires=[
        "opencv-python",
        "opencv-contrib-python>=4.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)