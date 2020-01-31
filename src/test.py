from nlp_takehome.src.cipher_take_home import generate_data

plain, cipher = generate_data(1<<20)

print(len(plain))

with open('data/enc-eng.txt', 'w') as f:
    for i in range(len(plain)):
        if i % 100 == 0:
            print(i)
        f.write(f"{cipher[i]}\t{plain[i]}\n")
