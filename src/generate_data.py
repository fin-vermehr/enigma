from nlp_takehome.src.cipher_take_home import generate_data

plain, cipher = generate_data(15000 * 16)

with open('../data/enc-eng.txt', 'a') as f:
    for i in range(len(plain)):
        f.write(f"{cipher[i]}\t{plain[i]}\n")
