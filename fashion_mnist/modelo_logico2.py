import scallopy

ctx = scallopy.ScallopContext(
    provenance="difftopkproofs",
    k=3
)

# Predicciones de la red
ctx.add_relation("digit1", int)
ctx.add_relation("digit2", int)
ctx.add_relation("digit3", int)

# Restricciones
ctx.add_relation("allowed_digit1", int)
ctx.add_relation("allowed_digit2", int)
ctx.add_relation("allowed_digit3", int)

# Resultado
ctx.add_relation("sum", int)
ctx.add_relation("valid", ())
ctx.add_relation("invalid", ())

# -------------------------
# Probabilidades de la red
# -------------------------

# prob1 = [0.02,0.05,0.99,0.08,0.03,0.01,0,0,0.01,0]
# prob2 = [0.01,0.99,0.02,0.03,0.02,0.01,0,0,0.01,0]
# prob3 = [0.01,0.02,0.03,0.04,0.05,0.90,0.02,0.10,0.01,0.02]

prob1 = [0.02,0.08,0.05,0.99,0.03,0.01,0,0,0.01,0]
prob2 = [0.01,0.01,0.02,0.03,0.02,0.99,0,0,0.01,0]
prob3 = [0.01,0.02,0.03,0.04,0.05,0.90,0.02,0.10,0.01,0.02]

ctx.add_facts(
    "digit1",
    [(prob1[i], (i,)) for i in range(10)]
)

ctx.add_facts(
    "digit2",
    [(prob2[i], (i,)) for i in range(10)]
)

ctx.add_facts(
    "digit3",
    [(prob3[i], (i,)) for i in range(10)]
)

# -------------------------
# Restricciones
# -------------------------

# --- REGLAS DE CATEGORIZACIÓN ---
# Mapeamos los IDs de Fashion-MNIST a los conceptos del profesor
ctx.add_rule("es_upper(x) = x == 0 or x == 2 or x == 4 or x == 6")
ctx.add_rule("es_lower(x) = x == 1")
ctx.add_rule("es_shoes(x) = x == 5 or x == 7 or x == 9")


# -------------------------
# Reglas
# -------------------------

# --- REGLAS DE ASIGNACIÓN ---
# Evaluamos qué categoría tiene cada una de las posiciones de la combinación
ctx.add_rule("tiene_upper() = (digit1(x) and es_upper(x)) or (digit2(x) and es_upper(x)) or (digit3(x) and es_upper(x))")
ctx.add_rule("tiene_lower() = (digit1(x) and es_lower(x)) or (digit2(x) and es_lower(x)) or (digit3(x) and es_lower(x))")
ctx.add_rule("tiene_shoes() = (digit1(x) and es_shoes(x)) or (digit2(x) and es_shoes(x)) or (digit3(x) and es_shoes(x))")

ctx.add_rule("outfit_valido() = tiene_upper() and tiene_lower() and tiene_shoes()")

ctx.run()

print("UPPER")
for p, t in ctx.relation("tiene_lower"):
    print(f"{t}, pb={p}")

print("VALID")
for p, t in ctx.relation("outfit_valido"):
    print(f"{t}, pb={p}")