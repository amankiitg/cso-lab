from setuptools import setup, find_packages

setup(
    name='cso-lab',
    version='0.1.0',
    description='Commodity Spread Option Analysis and Pricing Toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'panel>=0.14.0',
        'hvplot>=0.8.0',
        'scikit-learn>=1.0.0',
        'statsmodels>=0.13.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
