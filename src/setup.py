import os
import setuptools

readme_path = os.path.join("..", "README.md")
with open(readme_path, "r") as f:
    long_description = f.read()

setuptools.setup(
    name                            = "xor-orquestra",
    version                         = "0.1.0",
    author                          = "Zapata Computing, Inc.",
    author_email                    = "info@zapatacomputing.com",
    description                     = "XOR with a neural network in Orquestra.",
    long_description                = long_description,
    long_description_content_type   = "text/markdown",
    url                             = "https://github.com/luisguiserrano/xor-orquestra",
    packages                        = setuptools.find_packages(where = "python"),
    package_dir                     = {"" : "python"},
    classifiers                     = (
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires = [
        "torch",
        "numpy"
   ],
)
