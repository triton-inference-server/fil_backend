
import numpy as np
import scipy.special as sp


def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])


def print_c_array(name, X):
    formatted = ["{}".format(elem) for elem in np.nditer(X, order='C')]
    if len(X.shape) == 1:
        rows = [formatted]
    else:
        rows = [
            formatted[i : i + X.shape[1]] for i in range(0, len(formatted), X.shape[1])
        ]
    body = ",\n    ".join(["{" + ",".join(r) + "}" for r in rows])
    if len(X.shape) == 1:
        body = "const double {}[{}] = ".format(name, X.shape[0]) + body
    else:
        body = (
            "const double {}[{}][{}] = {{".format(name, X.shape[0], X.shape[1])
            + body
            + "}"
        )

    print(body+';')

# recursively subdivide the line adding nodes
def create_chebyshev_nodes(d):
    base = np.zeros(d)
    base[0] = -1.0
    base[1] = 1.0
    i = 2
    step = np.pi
    offset = -np.pi + step/2
    while i < d:
        base[i] = np.cos(offset)
        offset += step
        if offset >= 0.0:
            step /= 2
            offset = -np.pi + step/2
        i += 1

    return base


max_depth = 32

base = create_chebyshev_nodes(max_depth)
offset = np.vander(base + 1).T[::-1]
norm = np.zeros((max_depth + 1, max_depth))
for i in range(1, max_depth + 1):
    norm[i, :i] = np.linalg.inv(np.vander(base[:i]).T).dot(1.0 / get_norm_weight(i - 1))
print_c_array("kBase", base)
print_c_array("kOffset", offset)
print_c_array("kNorm", norm)