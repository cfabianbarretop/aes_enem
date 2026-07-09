import scallopy

ctx = scallopy.ScallopContext(
    provenance="difftopkproofs",
    k=3
)

# Predicciones de la red
ctx.add_relation("digit_1", int)
ctx.add_relation("digit_2", int)
ctx.add_relation("digit_3", int)
ctx.add_relation("upper", int)
ctx.add_relation("lower", int)
ctx.add_relation("shoe", int)

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
    "digit_1",
    [(prob1[i], (i,)) for i in range(10)]
)

ctx.add_facts(
    "digit_2",
    [(prob2[i], (i,)) for i in range(10)]
)

ctx.add_facts(
    "digit_3",
    [(prob3[i], (i,)) for i in range(10)]
)

# -------------------------
# Restricciones
# -------------------------

ctx.add_facts(
    "upper",
    [
        (1.0, (0,)),
        (1.0, (2,)),
        (1.0, (4,)),
        (1.0, (6,))
    ]
)

ctx.add_facts(
    "lower",
    [
        (1.0, (1,))
    ]
)

ctx.add_facts(
    "shoe",
    [
        (1.0, (5,)),
        (1.0, (7,)),
        (1.0, (9,))
    ]
)

# -------------------------
# Reglas
# -------------------------

ctx.add_rule("wear(X) :- digit_1(X)")
ctx.add_rule("wear(X) :- digit_2(X)")
ctx.add_rule("wear(X) :- digit_3(X)")

ctx.add_rule("has_upper(X) :- wear(X), upper(X)")
ctx.add_rule("has_lower(X) :- wear(X), lower(X)")
ctx.add_rule("has_shoe(X) :- wear(X), shoe(X)")

ctx.add_rule("valid() :- has_upper(U), has_lower(L), has_shoe(S)")

ctx.run()

print("SUMAS")
for p, t in ctx.relation("has_lower"):
    print(f"{t}, pb={p}")

print("VALID")
for p, t in ctx.relation("valid"):
    print(f"{t}, pb={p}")