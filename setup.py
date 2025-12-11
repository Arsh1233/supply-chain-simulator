from setuptools import setup, find_packages

setup(
    name="supply-chain-simulator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'networkx>=3.1',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.17.0',
        'scipy>=1.11.0',
        'gym>=0.26.0',
        'stochastic>=0.6.0',
    ],
    python_requires='>=3.8',
)