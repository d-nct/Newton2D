from rootfinding import newton
import numpy as np

# Para funções de 1 variável
def test1D():
    def f(x):
        return x**2 - 4

    raiz, numIter, trace = newton(f, x0=1, verbose=True)

    print("Raíz:", raiz)
    print("Número de iterações:", numIter)
    print("Trace:", trace)

# Para funções de 2 variáveis
def test2D():
    def g(x, y):
        return x**2 + y**2 - 4

    raiz, numIter, trace = newton2D(g, x0=1, y0=1, verbose=True)

    print("Raíz:", raiz)
    print("Número de iterações:", numIter)
    print("Trace:", trace)

if __name__ == "__main__":
    # test1D()
    test2D()
