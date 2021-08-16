from setuptools import setup

setup(
    name='PCTVAE',
    version='0.0.1',
    description="PCTVAE",
    author="T. Anderson Keller",
    author_email='',
    packages=[
        'pctvae'
    ],
    entry_points={
        'console_scripts': [
            'pctvae=pctvae.cli:main',
        ]
    },
    python_requires='>=3.6',
)