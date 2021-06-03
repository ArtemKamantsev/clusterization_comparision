import itertools

l = sorted([[1, 2], [2, 1]], key=lambda i: i[1])
print(list(itertools.islice(l, 1)))
