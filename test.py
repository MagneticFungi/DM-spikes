import sympy as sp

print("SymPy importado correctamente.")
print("Versión de SymPy:", sp.__version__)

# Prueba simbólica básica
x = sp.symbols("x")

expr = sp.sin(x)**2 + sp.cos(x)**2
expr_simplificada = sp.simplify(expr)

print("Expresión original:")
print(expr)

print("Expresión simplificada:")
print(expr_simplificada)

# Derivada simbólica
f = x**3 + 2*x**2 - 5*x + 1
df = sp.diff(f, x)

print("Función:")
print(f)

print("Derivada:")
print(df)

# Integral simbólica
integral = sp.integrate(sp.exp(-x), x)

print("Integral de exp(-x):")
print(integral)

# Verificación final
if expr_simplificada == 1 and df == 3*x**2 + 4*x - 5:
    print("Todo funciona correctamente.")
else:
    print("SymPy se importó, pero algo salió distinto en las pruebas.")
