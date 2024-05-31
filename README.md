# dr-rise
Multiscale Demand Forecasting

### Installation / upgrade
If some packages have been updated (pyforecaster), run 

`pip install -r /home/elettra/remote_projects/V2G4/requirements.txt --force-reinstall --upgrade`

this force a re-installation of all packages in the requirements.txt file, including `jax[cuda12_local]` which must be 
installed in a specific sequence (especially after jax)

NOTE: It could be needed to run the following:

`pip install --upgrade "jax[cuda12_local]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
Seems like the version 0.4.23 is the only one that works with the current version of jaxlib, see here
https://github.com/YoshitakaMo/localcolabfold/issues/212

`pip install --upgrade "jax[cuda12_local]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
and optax == 0.2.2 now works