from setuptools import setup, find_packages

setup(
    name='bcpipeline',
    version='0.1.0',
    author='Mazyad Alrashidi',
    author_email='mzydhaif11@gmail.com',
    description='Binary Classification Pipeline with feature engineering and ensemble methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mazyad-alrashidi/binary-classification-pipeline',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21',
        'pandas>=1.3',
        'scikit-learn>=1.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
