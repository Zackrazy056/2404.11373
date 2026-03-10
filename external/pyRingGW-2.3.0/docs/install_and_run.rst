Install and run
---------------

- Installing from `pip <https://pypi.org/project/pyRingGW>`_:
   
   .. code-block:: bash

      $ pip install pyRingGW
   
- Installing the `source code <https://git.ligo.org/lscsoft/pyring>`_:
   
   In your ``~/.bashrc`` add:  

   .. code-block:: bash

      $ export PYRING_PREFIX=/home/installation_directory/pyring  

   where ``installation_directory`` is the directory where you will be placing the ``pyRing`` source code. This path is needed for advanced functionalities such as QNM interpolation or injection of NR data.

   .. code-block:: bash

         $ git clone https://git.ligo.org/lscsoft/pyring.git 
         $ cd pyring
         $ git lfs install 
         $ git lfs pull  
         $ pip install -r requirements.txt
         $ python setup.py install  
   
   Add ``--user`` to the last command in case you don't have administrator's permissions (for example if you are installing on a cluster).    
   Alternatives to the last command are ``python -m pip install .`` or ``pip install .``  

- Running the code:

   The one-liner you are searching for is:
   
   .. code-block:: bash
   
      $ pyRing --config-file config.ini


- Examples:

   The ``config_files`` directory contains a variety of example files to analyse GW detections and injections (both ringdown templates and IMR injections). There is one example file for each waveform model supported, included all modifications to the Kerr hypothesis.
   The configuration files directory can either be found on the source code `repository <https://git.ligo.org/lscsoft/pyring/-/tree/master/pyRing/config_files>`_ or under the path ``installation_path/pyRing/config_files``. 
   To discover your ``installation_path``, type on the terminal:  

   .. code-block:: bash
   
      $ import pyRing
      $ pyRing.__file__

   that will output the path under which the package was built.
 
   A very fast example, to get you up to speed, can be launched by:
   
   .. code-block:: bash
     
      $ pyRing --config-file installation_path/pyRing/config_files/Quickstart_configs/quick_gw150914_DS.ini

   This allows the beginner to obtain a quick and rough measurement of GW150914 ringdown spectrum in under 3 minutes on a laptop (using aggressive priors and very light sampler settings).
   Other similar examples are available inside the `Quickstart_configs directory <https://git.ligo.org/lscsoft/pyring/-/tree/master/pyRing/config_files/Quickstart_configs>`_. These examples are mainly meant to give beginners a sense of the output of the code for various models.  
 
   A fast analysis, with settings more reliable than the above, can be obtained by launching:
   
   .. code-block:: bash
     
      $ pyRing --config-file repopath/pyRing/config_files/config_gw150914_local_data.ini

   This run still uses a simplified noise estimation and simplified sampler settings, but allows to obtain a decent measurement in ~20 minutes.

   Instead, a proper configuration file for the same run using production settings (hence obtaining publication-level results) can be launched by:

   .. code-block:: bash
   
      $ pyRing --config-file repopath/pyRing/config_files/config_gw150914_production.ini
   
   Never forget that the sampler settings may need adjustment based on the problem you want to tackle.
   See the `Usage` section for further discussion.

- Explore:

   The software supports a variety of analysis and injection options, all of which can be explored by running:

   .. code-block:: bash

      $ pyRing --help 

- Requirements:
 
   The software is guaranteed to be compatible with ``3.6=<python=<3.9``.
