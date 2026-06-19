from setuptools import setup, find_packages

setup(
    name             = "adve",
    version          = "2.0.0",
    author           = "Asmitha",
    author_email     = "asmitha2025@users.noreply.github.com",
    description      = "Anchor-Delta Video Embedding — efficient semantic video understanding",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/asmitha2025/ADVE",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "openai-clip",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu":      ["faiss-gpu>=1.7.4"],
        "training": ["scikit-learn>=1.3.0"],
        "dev":      ["pytest>=7.0", "black", "isort", "mypy", "httpx"],
    },
    entry_points={
        "console_scripts": [
            "adve=adve.core.main:main",
            "adve-server=adve.api.server:run",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    keywords=[
        "video understanding", "semantic embedding", "CLIP",
        "efficient inference", "object tracking", "spatial graph",
        "video AI", "edge deployment",
    ],
)
