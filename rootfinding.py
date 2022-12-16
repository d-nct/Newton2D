import numpy as np

def newton(f, x0: float=0, prec: float=1e-8, ytol: float=1e-8, maxiter: int=100, verbose: bool=False):
    """Aplica o método de Newton na função  f  e retorna a aprox. da raiz (com o erro mínimo erro_min).
    Parameters
    ----------
    f : function
        Função a qual será aplicada o Método de Newton
    x0 : float, opcional
        Ponto inicial a ser vasculhado. Idealmente, está próximo da raiz. O padrão é 0.
    prec : float, optional
        Tamanho do passo de newton pequeno o suficiente para retornarmos um valor. O padrão é 1e-8.
    ytol : float, optional
        Tamanho da tolerância ao redor do eixo y para retornarmos um valor. O padrão é 1e-8.
    maxiter : int, optional
        Máximo de iterações da função. O padrão é 100.
    verbose : bool, optional
        Estabelece se irá retornar, também, uma lista com os x_n alcansados.
    Returns
    -------
        Se verbose=True:
        <raíz: float>, <número de iterações: int>, <[x_1, x_2, x_3, ...]: list>
        Se verbose=False:
        <raíz: float>, <número de iterações: int>
    """
    # Definindo a derivada (central) de f
    df = lambda x: (f(x + prec) - f(x - prec)) / (2 * prec)

    trace = [x0]
    num_iter = 0
    x_i = x0

    while num_iter <= maxiter:
        try:
            passo = f(x_i)/df(x_i)
        except ZeroDivisionError:
            passo = f(x_i)/prec
        novo_x, num_iter = x_i - passo, num_iter + 1
        trace.append(novo_x)
        x_i = novo_x
        if   abs(passo)  < prec: break
        elif abs(f(x_i)) < ytol: break
        

    if num_iter > maxiter:
        novo_x = None # Retornamos None para alertar o usuário que não encontramos uma raiz
    if verbose: return novo_x, num_iter, np.array(trace)
    else: return novo_x, num_iter

def newton2D(f, x0: float=0, y0: float=0, prec: float=1e-8, ztol: float=1e-8, maxiter: int=100, verbose: bool=False):
    """Aplica o método de Newton na função  f  e retorna a aprox. da raiz (com o erro mínimo erro_min).
    Parameters
    ----------
    f : function
        Função a qual será aplicada o Método de Newton
    x0, y0 : float, opcional
        Ponto inicial a ser vasculhado. Idealmente, está próximo da raiz. O padrão é 0.
    prec : float, optional
        Tamanho do passo de newton pequeno o suficiente para retornarmos um valor. O padrão é 1e-8.
    ztol : float, optional
        Tamanho da tolerância ao redor do eixo y para retornarmos um valor. O padrão é 1e-8.
    maxiter : int, optional
        Máximo de iterações da função. O padrão é 100.
    verbose : bool, optional
        Estabelece se irá retornar, também, uma lista com os x_n alcansados.
    Returns
    -------
        Se verbose=True:
        <x: float>, <y: float>, <número de iterações: int>, <[(x_1,y_1), (x_2,y_2), (x_3,y_3), ...]: list>
        Se verbose=False:
        <x: float>, <y: float>, <número de iterações: int>
    """
    # Definindo as derivadas e o gradiente
    fx = lambda x, y: (f(x + prec, y) - f(x - prec, y)) / (2 * prec)
    fy = lambda x, y: (f(x, y + prec) - f(x, y - prec)) / (2 * prec)
    grad = lambda x, y: np.array([fx(x, y), fy(x, y)])

    # Definindo as variáveis auxiliares
    trace = [x0]
    num_iter = 0
    x_i = x0

    # TODO: Adaptar essa parte para 2D
    while num_iter <= maxiter:
        try:
            passo = f(x_i)/df(x_i)
        except ZeroDivisionError:
            passo = f(x_i)/prec
        novo_x, num_iter = x_i - passo, num_iter + 1
        trace.append(novo_x)
        x_i = novo_x
        if   abs(passo)  < prec: break
        elif abs(f(x_i)) < ytol: break
        

    if num_iter > maxiter:
        novo_x = None # Retornamos None para alertar o usuário que não encontramos uma raiz
    if verbose: return novo_x, num_iter, np.array(trace)
    else: return novo_x, num_iter
