from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='pytriton-inference-server',
    version='0.1',
    description='Serves model inference requests.',
    long_description=readme(),
    url='https://github.com/neurobooth/triton',
    author='Brandon Oubre',
    author_email='boubre@mgh.harvard.edu',
    license='BSD 3-Clause License',
    packages=['inference_server'],
    include_package_data=True,
    install_requires=[],
)
