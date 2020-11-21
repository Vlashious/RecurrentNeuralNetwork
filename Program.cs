// Written by Uładzimir Śniežka, 821701
// Used libraries are Math.NET.

using System;
using RecursiveNetwork;

TargetMatrix m = new TargetMatrix();
NN nn = new NN(m.Matrix);
nn.Train();