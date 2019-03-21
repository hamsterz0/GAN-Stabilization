values = []

filename = 'values-1'
with open(filename, 'r') as f:
    for line in f:
        line = line.split(',')

print(len(line))
print(line)

loss, acc, i = [], [], 0

for value in line:
    if i % 2 == 0:
        loss.append(value)
    else:
        if len(value) > 4:
            acc_val = value[:4]
            loss_val = value[4:]
            loss.append(loss_val)
        else:
            acc_val = value
        acc.append(acc_val)
    i += 1

print(len(loss))
print(len(acc))

with open('fix-values-1', 'w') as f:
    for l, a in zip(loss, acc):
        f.write('{},{}\n'.format(l,a))


