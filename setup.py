import setuptools

setuptools.setup(
    name="multisys_pipeline",

    packages=setuptools.find_packages(),
)


# If there is a warning "no module named multisys_pipeline" then place setup.py in the folder multisys_pipeline
# cd to the folder multisys_pipeline, e.g. >> cd /multisys_pipeline
# >> pip install -e .
# Then restart the integrated development environment (IDE)

# Alternatively, cd to the folder multisys_pipeline, e.g. >> cd /multisys_pipeline
# >> pip install -r setup_all.txt
