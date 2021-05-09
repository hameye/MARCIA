from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().strip().split('\n')


setup(
    name='MARCIA',
    version='0.1.3',
    description='Manual hyperspectral data classifier',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hameye/marcia',
    author='Hadrien Meyer',
    author_email='meyerhadrien96@gmail.com',
    license='GPL v3',
    packages=find_namespace_packages(
        exclude=[
            'doc',
            'doc.*']),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Typing :: Typed",
    ],
    python_requires='>=3.6',
    install_requires=requirements(),
    zip_safe=False,
    project_urls={
        'Example': 'https://github.com/hameye/MARCIA/blob/master/examples/Tutorial.ipynb',
        'Source': 'https://github.com/hameye/MARCIA',
        'Report a bug': 'https://github.com/hameye/MARCIA/issues',
    })
