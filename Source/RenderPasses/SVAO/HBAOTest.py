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


def hbao_kernel(r, y):
    if y == 0:
        return 0
    return max(0, 1.0 - r * r - y * y) * max( y / math.sqrt(r * r + y * y) - 0.1, 0)

def pdf(r):
    return pow(1 - r, 1.5) * 0.9

numeric = []
analytic = []

for r in range(100):
    max_val = 0
    for y in range(100):
        max_val = max(hbao_kernel(r/100, y/100), max_val)
    
    numeric.append(max_val)
    analytic.append(pdf(r/100))
    #print(f"r: {r/20}, max_val: {max_val}, pdf: {pdf(r/20)}")

# plot numeric and analytic in pyplot
import matplotlib.pyplot as plt
plt.plot(numeric)
plt.plot(analytic)
# add legend
plt.legend(["Numeric", "Analytic"])
plt.show()
