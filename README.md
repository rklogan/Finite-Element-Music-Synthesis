# Finite-Element-Music-Synthesis

Finite Element Music Synthesis is a collection of programs that simulate the motion of a drum skin for the purpose of audio synthesis. The programs were used to compare the effects of GPU parallelization schemes when applied to the given synthesis algorithm.

## q1.py

A sequential simulation. Used as a baseline.

## q2.py

A naive parallelization scheme whereby each point on the membrane is processed by a different thread.

## q3.py

Implements the synthesis algorithm using the finite element method whereby threads are assigned a subset of points on the membrane that are locally clustered to improve cache performance.

## License

MIT License

Copyright (c) [2019] [Ryan Khan Logan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
