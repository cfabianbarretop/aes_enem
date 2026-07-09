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

prob1 = [0.02,0.05,0.99,0.08,0.03,0.01,0,0,0.01,0]
prob2 = [0.01,0.99,0.02,0.03,0.02,0.01,0,0,0.01,0]
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

ctx.add_facts(
    "allowed_digit1",
    [
        (1.0, (0,)),
        (1.0, (2,)),
        (1.0, (4,)),
        (1.0, (6,))
    ]
)

ctx.add_facts(
    "allowed_digit2",
    [
        (1.0, (1,))
    ]
)

ctx.add_facts(
    "allowed_digit3",
    [
        (1.0, (5,)),
        (1.0, (7,)),
        (1.0, (9,))
    ]
)

# -------------------------
# Reglas
# -------------------------

ctx.add_rule("""
sum(S) :-
    digit1(A),
    allowed_digit1(A),
    digit2(B),
    allowed_digit2(B),
    digit3(C),
    allowed_digit3(C),
    S == A + B + C
""")

ctx.add_rule("""
valid() :-
    sum(S),
    S >= 6,
    S <= 12,
    S % 2 == 0
""")

ctx.run()

print("SUMAS")
for p, t in ctx.relation("sum"):
    print(f"{t}, pb={p}")

print()

print("VALID")
print(list(ctx.relation("valid")))