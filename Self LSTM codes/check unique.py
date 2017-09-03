data = open('umbrella.txt').read()

print(list(set(data.split())))
