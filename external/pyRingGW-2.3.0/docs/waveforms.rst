Ringdown models
-------------------------

pyRing supports most of the analytical ringdown models available in the literature:

- Damped Sinusoids model

    This is the simplest model available, consisting of a superposition of damped sinusoid . It supports up to all five polarisations 
    (the ones independent in a L-shaped GW detector, see `this reference <https://arxiv.org/abs/1710.03794>`_) allowed for a metric theory of gravity, employing the conventions:

    .. math::
        \begin{aligned}
        h_s &= \sum_{j} A^s_j \cdot cos(\omega^s_j t+\phi^s_j)  \cdot e^{-(t-t^0_j)/\tau^s_j} \, ,\\
        h_{v_x} - i h_{v_y} &= \sum_{j} A^v_j \cdot e^{i( \omega^v_j t+\phi^v_j)} \cdot e^{-(t-t^0_j)/\tau^v_j} \, ,\\
        h_{+}  - i h_{\times}  &= \sum_{j} A^t_j \cdot e^{i( \omega^t_j t+\phi^t_j)} \cdot e^{-(t-t^0_j)/\tau^t_j} \, .
        \end{aligned}

    Where, for each mode, the complex amplitudes and complex frequencies :math:`A_j, \phi_j, \omega_j, \tau_j` are left free to vary and :math:`t^0_j` is the start time.

- Kerr model

    .. math::

        h_+ + i h_{\times} = \frac{M_f}{D_L} \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^{+\ell} \sum_{n=0}^{\infty}\, \, (h^{+}_{\ell m n} + h^{-}_{\ell m n})

    with:

    .. math::

        \begin{aligned}
        h^{+}_{\ell m n} &= \mathcal{A}^{+}_{\ell m n} \, S_{\ell m n}( \iota, \varphi) \, e^{i[(t-t_{\ell m n})\tilde{\omega}_{\ell m n}+\phi^{+}_{\ell m n}]} \\
        h^{-}_{\ell m n} &= \mathcal{A}^{-}_{\ell m n} \, S_{\ell -m n}(\iota, \varphi) \, e^{-i[(t-t_{\ell m n})\tilde{\omega}^*_{\ell m n}-\phi^{-}_{\ell m n}]} 
        \end{aligned}


    where :math:`\tilde{\omega}_{\ell m n} = {\omega}_{\ell m n} + i/{\tau_{\ell m n}}` is the complex ringdown frequency (a * denotes complex conjugation), 
    expressed as a function of the remnant BH mass :math:`M_f` and spin :math:`a_f`, :math:`\tilde{\omega}_{\ell m n} = \tilde{\omega}_{\ell m n}(M_f, a_f)`.
    The amplitudes :math:`\mathcal{A}^{+/-}_{\ell m n}` and phases :math:`\phi^{+/-}_{\ell m n}` characterise the excitation of each mode. 
    In the case where the progenitors binary has spins aligned with the orbital angular momentum, reflection symmetry implies (see e.g. `Blanchet's review <https://arxiv.org/abs/1310.1528>`_):
    :math:`\tilde{A}^{-} = (-1)^l \cdot {\tilde{A}^{+}}^*`, halving the number of free parameters per mode (* denotes complex conjugation, and we grouped the :math:`\mathcal{A}, \phi` parameters in a complex amplitude).
    The inclination of the BH final spin relative to the observer's line of sight is denoted by :math:`\iota`, 
    while :math:`\varphi` corresponds to the azimuthal angle of the line of sight in the BH frame. 
    :math:`S_{\ell m n}` are the spin-weighted `spheroidal harmonics <https://arxiv.org/abs/1408.1860>`_ and :math:`t_{\ell m n}=t_0` is a reference start time.
    In writing these equations, we follow the conventions of `Lim et al. <https://arxiv.org/abs/1901.05902>`_ (see their Section III), 
    accoding to which :math:`m>0` indices denote co-rotating modes, while counter-rotating modes are labeled by :math:`m<0`. 
    Counter-rotating modes are hardly excited in the post-merger phase for the binaries observed by LIGO-Virgo-Kagra. 
    For a discussion about the possible relevance of counter-rotating modes see Refs. `[1] <https://arxiv.org/abs/2010.08602>`_, `[2] <https://arxiv.org/abs/1901.05902>`_.
    For applications of this model to numerical relativity simulations, see e.g. `2110.03116 <https://arxiv.org/abs/2110.03116>`_, `2302.03050 <https://arxiv.org/abs/2302.03050>`_.

    This model additionally supports:

    - Tail terms, see `arxiv:2302.03050 <https://arxiv.org/abs/2302.03050>`_ and references therein.
  
    - Quadratic modes, see `arxiv:2208.07374 <https://arxiv.org/abs/2208.07374>`_ and references therein.

    - Parametric deviations from GR QNM predictions:

        Parametrised deviations to QNM frequencies and damping times as predicted by GR can be added in the form:

        .. math::

            \begin{aligned}
            \omega_{lmn} &= \omega^{GR}_{lmn} \cdot (1+\delta\omega_{lmn})\, ,\\
            \tau_{lmn} &= \tau^{GR}_{lmn} \cdot (1+\delta\tau_{lmn})\, .\\
            \end{aligned}

        and extracted from the data. See the `GWTC-2 testing GR catalog <https://arxiv.org/abs/2010.14529>`_ for additional details on the method.

    - QNM spectrum induced by the area quantisation:

        The ringdown model from `Foit and Kleban <https://arxiv.org/abs/1611.07009>`_, assuming a modification of the ringdown spectrum induced by the
        area quantisation hypothesis is also implemented. The quantisation is parametrised by a new parameter :math:`\alpha`,
        determining the new QNM spectrum, labeled by an index :math:`N`:

        .. math::

            \begin{aligned}
            \omega_N &= \omega_N(M_f, a_f, \alpha)\, ,\\
            \tau_N &= \tau_N(M_f, a_f, \alpha)\, .\\
            \end{aligned}

        For details of the implementation, the application of the model to current observations and 
        to simulated data, see `this reference <https://arxiv.org/abs/2011.03816>`_.
        A more realistic model of this scenario, considering the echoes generated from such a modification,
        was presented in `this reference <https://arxiv.org/abs/1902.10164>`_. 
        For an extended discussion, see `Agullo et al. <https://arxiv.org/abs/2007.13761>`_.

    - ParSpec deviations:

        The software also supports perturbative deviations within the `Parametrized ringdown spin expansion coefficients` formalism,
        introduced in `this reference <https://arxiv.org/abs/1910.12893>`_.
        The formalism first considers a perturbative expansion of the QNM spectrum in powers of the remnant spin,
        adding deviation parameters at each given order in spin, in the form:

        .. math::

            \begin{aligned}
            \omega_K &= \frac{1}{M} \, \sum_{j=0}^{N_{max}} \, \chi^j \, \omega^{(j)}_K \, (1+\gamma \, \delta \omega_K^{(j)})\, ,\\
            \tau_K &= M \, \sum_{j=0}^{N_{max}} \, \chi^j \, \tau^{(j)}_K \, (1+\gamma \, \delta \tau_K^{(j)}) \, .
            \end{aligned}

        where each mode is labeled by :math:`K`, :math:`\omega^{(j)}_K, \tau^{(j)}_K` are the GR spin-expansion coefficients and
        :math:`\delta \omega_K^{(j)}` are the beyond-GR coefficients, to be inferred from the data. The beyond GR coupling is controlled by the parameter:

        .. math::
        
            \gamma = \frac{\alpha \,(1+z)^p}{M_f^p}
    
        expressed as a function of a theory-dependent coupling :math:`\alpha`, with :math:`p` representing its mass dimension.

        Such a formalism encompasses large classes of modified theories of gravity. Depending on the theory 
        considered, in certain cases it allowed to place the most stringent constraints to date on some of
        these alternative theories. 
        Details of the implementation and the application to observational data were presented `here <https://arxiv.org/abs/2102.05939>`_.

    - Eistein-scalar-Gauss-Bonnet corrections:

        Coming soon...

    - Kerr-Newman charges:

        Coming soon...

    Note: at the moment, for the Kerr multipolar model, the modes are supposed to start all at the same time. 
    This implicitly assumes that all the modes are already excited when the analysis is start.

- Multi-modal ringdown non-spinning (MMRDNS) model

    This model, introduced in `this reference <https://arxiv.org/abs/1404.3197>`_ is an improvement of the Kerr model in the case 
    where the remnant black hole is generated by the quasi-circular coalescence of two non-spinning progenitor black holes.
    It models the most dominant modes (up to :math:`\ell=5`) for the parameter space considered, assumes the conjugate symmetry discussed above
    and does not keep into account counter-rotating modes.
    The amplitudes and phases are tuned to BBH numerical simulations and are expressed as a function of the progenitors parameters:
    
    .. math::

        \begin{aligned}
        \mathcal{A}_{lmn} = \mathcal{A}_{lmn}(\eta)\, ,\\
        \phi_{lmn} = \phi_{lmn}(\eta)\, .\\
        \end{aligned}

    where :math:`\eta` is the symmetric mass ratio of the progenitors binary.
    The model describes only the late ringdown and was calibrated at :math:`10 M_f` after the peak of :math:`\psi^{NR}_{22}`.
    For low-SNR events it can be extrapolated to earlier times, but its accuracy should be explicitly checked.
    See also `this reference <https://arxiv.org/abs/1805.04760>`_ for a discussion of the start time and an application of the model to ringdown parameter estimation.

- Multi-modal ringdown non-precessing (MMRDNP) model

    This model, introduced in `this reference <https://arxiv.org/pdf/1801.08208.pdf>`_, is an improvement to MMRDNS to the case of spinning, non-precessing, progenitors.
    It employs a spherical decomposition, keeping into account mode mixing between different spheroidal modes.
    It models the most dominant modes (up to :math:`\ell=4`) for the parameter space considered, assumes the conjugate symmetry discussed above
    and does not keep into account counter-rotating modes.
    The complex amplitudes are now expressed as:
    
    .. math::

        \begin{aligned}
        \mathcal{A}_{lm} = \mathcal{A}_{lm}(\eta, \chi_s, \chi_a)\, ,\\
        \phi_{lm} = \phi_{lm}(\eta, \chi_s, \chi_a)\, .\\
        \end{aligned}

    where :math:`\eta` is the symmetric mass ratio of the progenitors binary, :math:`\chi_s` is a symmetric spin combination of the progenitors binary, 
    and :math:`\chi_a` is a anti-symmetric spin combination of the progenitors binary.
    The model describes only the late ringdown and was calibrated at :math:`20 M_f` after the peak of :math:`h_{22}`.
    For low-SNR events it can be extrapolated to earlier times, but its accuracy should be explicitly checked.

- Kamaretsos-Hannam-Husa-Sathyaprakash (KHS) multi-modal model

    This model, introduced in `[1] <https://arxiv.org/abs/1207.0399>`_ and `[2] <https://arxiv.org/abs/1406.3201>`_ is similar to MMRDNP, 
    but uses a different fit of the amplitudes. Phases have not been tuned to numerical relativity and are set to zero by default.
    It models the most dominant modes (up to :math:`\ell=4`) for the parameter space considered, assumes the conjugate symmetry discussed above
    and does not keep into account counter-rotating modes.

- Tidal Effective One Body post-merger (TEOBPM) model

    The TEOBPM model, introduced in `Damour and Nagar <https://arxiv.org/abs/1406.0401>`_, accurately describes the whole post-merger phase 
    (from the peak of :math:`h_{22}` onwards) for spinning, non-precessing binaries,
    modeling through a spheroidal decomposition the most dominant modes (up to :math:`\ell=5`) 
    with a resummation strategy which keeps into account the overtones contributions.
    Similarly to previous models, amplitudes and phases are calibrated to full numerical simulation;
    crucially, since the model starts at the peak of the emission, in this case also the time-delays between the peak of the different modes are kept into account.
    Further details on the model can be found in references `[1] <arXiv:1606.03952>`_ and `[2] <arXiv:2001.09082>`_.

    This model also forms the basis for the ringdown of the `IMRPhenomTPHM model <https://arxiv.org/abs/2105.05872>`_.

    An implementation of the model in C is available `here <https://bitbucket.org/eob_ihes/teobresums/src/master/C/src/>`_.

A script showing how to handle all available models can be found in ``pyring/scripts/Waveform_utils/plot_waveform.py``.
