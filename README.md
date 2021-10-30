Dependencies: `CGAL 4.7`

Installation:

```
clone git@github.com:li9i/x1smsm-public.git
cd x1smsm-public
mkdir build
cmake ..
make
```

Execute tests with

```
 ./smsm_node 2 1 0 778 0.2 0.786 0.01 0.0 0 0 360 180 SKG censi
 ./smsm_node A B C   D   E     F    G   H 0 0   K   L SKG N
```

where

```
A: number of iterations of the tranlational component (2)
B: number of iterations over a specific dataset
C: start sample of the dataset
D: end sample of the dataset
E: maximal pose estimate translational displacement (-E,+E) m
F: maximal pose estimate rotational displacement (-F,+F) rad
G: sd of normally distributed noise added to real scan measurements (m)
H: sd of normally distributed noise added to each coordinate of the map (m)
K: size of the real scan
L: size of the map
N: dataset signifier; acceptable are `aces`, `fr079`, `intel`,`mit_csail`, `mit_killian`
```
