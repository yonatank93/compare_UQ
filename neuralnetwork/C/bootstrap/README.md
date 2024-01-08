Workflow:
1. First, run `bootstrap_generate_configs.py` to generate bootstrap configurations.
2. After that, train the model using each bootstrap dataset. The `bootstrap_single_set.py`
   can be used to run the training for a single bootstrap dataset. To iteratively submit
   training job for each bootstrap dataset to the supercumputer, run `bootstrap_submit_job.py`.
3. Then, run `bootstrap_install_uninstall_model.py` to write the KIM model for each sample
   and install the model.
4. As an option, run `bootstrap_archive_saved_models.py` to compress the saved model into
   a tar file. This process is helpful to reduce the number of files, e.g., if there is a
   quota for the number of files in supercomputer.
