cdef double _sym_mass_ratio(double m1, double m2) nogil
cdef double _X_1(double m1, double m2) nogil
cdef double _X_2(double m1, double m2) nogil
cdef double _X_12(double X1, double X2) nogil
cdef double _a_0(double X1, double X2, double chi1, double chi2) nogil
cdef double _a_12(double X1, double X2, double chi1, double chi2) nogil
cdef double _S_hat(double X12, double a0, double a12) nogil
cdef double _S_bar(double X12, double a0, double a12) nogil
cdef double _alpha1(double af, int l, int m) nogil
cdef double _alpha21(double af, int l, int m) nogil
cdef double _omega1(double af, int l, int m) nogil
cdef double _c3_A(double nu, double X12, double S_hat, double a12, int l, int m) nogil
cdef double _c3_phi(double nu, double X12, double S_hat, int l, int m) nogil
cdef double _c4_phi(double nu, double X12, double S_hat, int l, int m) nogil
cdef double _dOmega(double omega1, double Mf, double omega_peak) nogil
cdef double _amplitude_peak(double nu, double X12, double S_hat, double a12, double S_bar, double a0, double omega_peak, int l, int m) nogil
cdef double _omega_peak(double nu, double X12, double S_hat, double a0, int l, int m) nogil
cdef double _JimenezFortezaRemnantMass(double nu, double X1, double X2, double chi1, double chi2, double M) nogil
cdef double _JimenezFortezaRemnantSpin(double nu, double X1, double X2, double chi1, double chi2) nogil
cdef double _DeltaT(double nu, double X12, double S_hat, double a0, unsigned int l, int m) nogil
cdef double _DeltaPhi(double nu, double X12, double S_hat, unsigned int l, int m) nogil
