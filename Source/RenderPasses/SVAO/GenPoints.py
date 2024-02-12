import math

def van_der_corput(n, base=2):
    result = 0
    denom = 1
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result

# for n = 8
rngs = [van_der_corput(i) for i in range(8, 16)]
# for n = 16
rngs = [van_der_corput(i) for i in range(16, 32)]
# for n = 32
rngs = [van_der_corput(i) for i in range(32, 64)]
for i in rngs:
    print(i, end=", ")
print()

# VAO distribution sqrt(1-rng^(2/3))
print("VAO distribution: [", end="")
for rng in rngs:
    print((1-rng**(2/3))**0.5, end=", ")
print("]")

# HBAO distribution 2 * asin(sqrt(rng^1.25)) / pi
print("HBAO distribution: [", end="")
for rng in rngs:
    print(2 * math.asin((rng)**1.25) / math.pi, end=", ")
print("]")