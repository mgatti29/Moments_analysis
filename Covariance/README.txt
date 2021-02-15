you need to run, in order:

- compute_survey_mask.py : it saves the DES y3 mask to pkl format and for a given nside.
- convert_flask2maps.py : if a bunch of flask mocks is provided, it converts them into a format suitable for computing moments.
- run_flask_measurements.py : it runs the measurement on flask mocks.
- Compute_micing_matrixes.ipynb : notebook to compute the mixing matrixes.