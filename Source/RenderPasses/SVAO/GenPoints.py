import math

# van der coprut sequence from 8 to 15
rngs = [1, 9, 5, 13, 3, 11, 7, 15]
# divide each by 16
rngs = [x/16 for x in rngs]

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