# Cryopython
Home-made python based cryoEM package. Rather for learning than real structural biology purposes.

For now, it has simple skimage autopick functionality with export of found particles. It works surprisingly well!
There are also some quick tools to open and write MRC files, check the .mrcs files and histogram adjustments for high dynamic range MRC files. 
I was also trying to implement easy CTF correction, so you can do the power spectrum of micrograph, average pixels over different radii, and plot intensities dependence of a distance from center. CTF fitting function is there but it needs still more work.

CTF estimation:
![CTF fitting (kind of)](https://github.com/dzyla/Cryopython/blob/master/myplot.png)

Autopicking with preselected class:
![Super simple class selection](https://github.com/dzyla/Cryopython/blob/master/myplot1.png)
Image of picked particles and cross-correlation map:
![Scikit-image based cross-corellation autopick](https://github.com/dzyla/Cryopython/blob/master/myplot2.png)

# Addtional tools

MRC map 3D to 2D averaging with at all axes:
![average3D2D](https://github.com/dzyla/Cryopython/blob/master/2D1.JPG)

![average3D2D](https://github.com/dzyla/Cryopython/blob/master/2D2.JPG)

![average3D2D](https://github.com/dzyla/Cryopython/blob/master/2D3.JPG)
