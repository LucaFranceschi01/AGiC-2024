Group members:
    - Luca Franceschi (u199149)
    - Telmo Linacisoro (u198711)

We have tried to implement two detectors: the OpenCV haar-cascade detector and 
the DLIB MMOD detector with CNN-based features.

First we tried implementing the first one, which was kind of straight-forward,
and after tweaking a bit the parameters, we have achieved decent results in a 
very little amount of time.

The second one was a bit trickier since running it needs quite a lot of memory 
and uses GPU parallelization (which is transparent to us, but did not work well 
with our machines). It did look more promising than the first one, but we could 
not get it running in a decent amount of time (more than a minute per image).

For both of them what we have to do is quite similar: inside of the function 
MyFaceDetectionFunction we call these detectors and for each face detected, add 
the bounds to the matrix that we will eventually return.

We ended up delivering the first detector even though we spent more time trying 
to build the second one because we could not get it running.

The requirements for the python environment are described in "requirements.txt".

