from setuptools import setup, find_packages

setup(
    name='whale-call-classifier',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A classifier for whale calls using neural networks.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # or 'torch' if using PyTorch
        'matplotlib',
        'seaborn',
        'librosa',
        'jupyter',  # for running notebooks
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)