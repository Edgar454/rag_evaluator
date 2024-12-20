from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="rag-evaluator",          
    version="0.1.0",              
    author="MEVA'A EDGAR",            
    author_email="mevaed4@gmail.com",  
    description="A utility for evaluating Retrieval-Augmented Generation models.",
    url="https://github.com/edgar454/rag_evaluator",  
    packages=find_packages(),     
    install_requires=open("requirements.txt").readlines(),  
    include_package_data=True,
    package_data={
        "rag_evaluator": ["data/*.csv", "data/*.pkl"],  # Paths relative to the `rag_evaluator` package
    },
    python_requires=">=3.8",       
)

