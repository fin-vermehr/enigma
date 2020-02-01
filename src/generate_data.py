from nlp_takehome.src.cipher_take_home import generate_data

plain, cipher = generate_data(2000000)

with open('../data/enc-eng.txt', 'w') as f:
    for i in range(len(plain)):
        f.write(f"{cipher[i]}\t{plain[i]}\n")
