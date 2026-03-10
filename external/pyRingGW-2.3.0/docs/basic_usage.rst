Usage
-------------------


- Code input/output:

   The adopted configuration and the progress of the run are shown on the screen if the option ``screen-output=1`` is added inside the ``[input]`` section of the configuration file. Otherwise, the output from pyRing is stored in the ``example_outdir/stderr_pyRing.txt``, ``example_outdir/stdout_pyRing.txt`` files, while the progress of the sampler is stored inside the ``Nested_sample/cpnest.log`` file. Data and PSD diagnostic plots are produced at the beginning of the run and can be found in the ``example_outdir/Noise`` directory.
   All paths in the configuration files are relative to the path stored in the ``PYRING_PREFIX``.
   
   At the end of the run, the posterior and evidence files can be found in the ``example_outdir/Nested_sampler`` directory under ``posterior.dat`` and ``Evidence.txt``. Results on all parameters are displayed in the ``example_outdir/Plots`` directory. If the pesummary option is activated (see the help message for details), a PESummary metafile containing all the necessary information to reproduce the run, together with the complete run output, will be produced. Also, an html page displaying parameters posteriors will be created under: ``example_outdir/Plots/pesummary_postproc/home.html``.

- Sampler convergence:

   For details on the sampler used, ``CPNest``, see `CPNest documentation <https://johnveitch.github.io/cpnest/>`_. 
   
   Standard templates with a small number of parameters (e.g. a single damped sinusoid or a single Kerr mode) can be safely employed with the sampler settings documented in the ``repopath/pyRing/config_files/config_gw150914_production.ini`` example. 
   
   When assuming many Kerr modes (e.g. more than two overtones), care must be taken in assessing the sampler's convergence. To produce publication level results in these cases, we suggest checking that more conservative sampler settings such as ``nlive=8192``, ``maxmcmc=8192`` do not change the obtained results. We also suggest combining at least four parallel chains.
