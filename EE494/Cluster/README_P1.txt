Copyright (c) 2017, Tampere University of Technology (TUT)
This work is licensed under the MIT license / X11 license.
This documentation is licensed under CC0 license.


This folder contains the following files, written in Python

 clustering+positioning.py
 visualizeRssData.py


The two scripts written in Python, clustering+positioning.py and visualizeRssData.py, they
visualize and analyze the data provided in/by <ref to repo>.
    1) clustering+positioning.py - this file contains two positioning algorithms as benchmark
        for the provided data. They are based on the paper 'Clustering benefits in
        mobile-centric WiFi positioning in multi-floor buildings' by A. Cramariuc
        (DOI: 10.1109/ICL-GNSS.2016.7533846).
    2) visualizeRssData.py - this file visualizes the Test or the Training data analogously
        to the m-file visualizeRssData.m found in M1 sub-folder
The software was tested with Python 3.4.5. and depends the modules numpy,
scipy, matplotlib and scikit-learn.


Request for citation:

[1] E.S. Lohan, J. Torres-Sospedra, P. Richter, H. Lepp�koski, J. Huerta, A. Cramariuc, �Crowdsourced WiFi-fingerprinting database and benchmark software for indoor positioning�, Zenodo repository, DOI 10.5281/zenodo.889798
[2] A. Cramariuc, H. Huttunen and E. S. Lohan, "Clustering benefits in mobile-centric WiFi positioning in multi-floor buildings," 2016 International Conference on Localization and GNSS (ICL-GNSS), Barcelona, 2016, pp. 1-6.
doi: 10.1109/ICL-GNSS.2016.7533846