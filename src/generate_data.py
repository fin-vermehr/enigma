from cipher_main import generate_data

plain, cipher = generate_data(30000 * 16)

with open('data/enc-eng.txt', 'a') as f:
    for i in range(len(plain)):
        f.write(f"{cipher[i]}\t{plain[i]}\n")
