from setuptools import setup, find_packages

setup(
    name='career_guide_prediction',
    version='0.1.0',
    description='Career Guide Prediction System',
    author='Tushar Chaudhary',
    author_email='chaudharytushar477@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'Flask>=3.1.1',
        'scikit-learn>=1.7.0',
        'pandas>=2.3.0',
        'numpy>=1.24.0',
        'python-dotenv>=1.0.0',
        'joblib>=1.2.0'
    ],
    entry_points={
        'console_scripts': [
            'run-training=src.run_training:main',
            'run-preprocessing=src.data_preprocessing:main'
        ]
    }
)
