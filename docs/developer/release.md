# Creating a new release

Steps for creating a new release:

1. Update the version number in the `powertrain/__init__.py` file.

2. (Optional) Train a new set of models if there are new features that models would benefit from or if the model api has been fundamentally changed.
    - This is the current script to train the models: `/projects/mbap/nreinick/repos/routee-validation-report/py-notebooks/train_all_models.ipynb`.

    - Upload the models to a new sub folder in [this box folder](https://app.box.com/s/v2rm6b35f0b2hh9dbco82gvrt9ihgylq). You can use the `box-upload` command from [pyeagle](https://app.box.com/s/v2rm6b35f0b2hh9dbco82gvrt9ihgylq) 
    
    - Copy a representative ICE and BEV model into `powertrain/resources/default_models/` and update the `local_models` object in `powertrain/io/load.py` if necessary. 

    - Build a new set of external model links using the script `scripts/developers/build_box_shared_links.py`.

3. Open a PR with the new version.

4. Build the package using `python setup.py sdist bdist_wheel`.

5. Add the new package to the [mbap-pypi](https://github.nrel.gov/MBAP/mbap-pypi) index.

